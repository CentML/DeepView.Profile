import subprocess
import sys
import os
import sqlite3
from collections import defaultdict

NS_TIME = 1e9

def register_command(subparsers):
    parser = subparsers.add_parser(
        "gpu-usage-estimation",
        help="Estimation of gpu operation time. We recommend running your training cycle for 100 iterations"
    )

    parser.add_argument(
        "path_to_file",
        help="path of the file you want to analyze"
    )
    parser.set_defaults(func=main)

def joinIntervals(arr):
    # arr = tuple(type,start,end,streamid)
    eventDict = defaultdict(int)
    filteredArr = []
    prevRecord = list(arr[0])
    for i in range(1,len(arr)):
        newRecord = list(arr[i])
        if prevRecord[1] <= newRecord[1] <= prevRecord[2]:
            prevRecord[1] = min(prevRecord[1], newRecord[1])
            prevRecord[2] = max(prevRecord[2], newRecord[2])
        else:
            filteredArr.append(prevRecord)
            prevRecord = newRecord
    filteredArr.append(prevRecord) # last record does not enter the validation cycle. has to be included at the end
    for item in filteredArr:
        eventDict[item[0]] += (item[2]-item[1])
    return eventDict

def sql_command_execution(db_path):
    connection = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES)
    cursor = connection.cursor()

    try:
        timeline_data = cursor.execute("""
            SELECT "memOps" as name, start,end, streamId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            UNION ALL
            SELECT "kernelOps" as name, start, end, streamId
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            UNION ALL
            SELECT "memOps" as name, start, end, streamId
            FROM CUPTI_ACTIVITY_KIND_MEMSET
            ORDER by start ASC;
            """).fetchall()
        profiling_duration = cursor.execute("""
            SELECT duration FROM ANALYSIS_DETAILS;
            """).fetchone()[0]
        cupti_api_duration = cursor.execute("""
            SELECT max(end)-min(start) from CUPTI_ACTIVITY_KIND_RUNTIME;
            """).fetchone()[0]
        cursor.close()
    except sqlite3.Error as er:
        print("There was an error reading the information from the sqlite database")
        print('SQLite error: %s' % (' '.join(er.args)))
        cursor.close()
        sys.exit(1)

    if not timeline_data:
        print("There are no traces of gpu activity")
        sys.exit()
    gpu_activity_time = joinIntervals(timeline_data)
    percgpu_activity = ((gpu_activity_time["kernelOps"]+gpu_activity_time["memOps"])/cupti_api_duration)*100
    data = [round(profiling_duration/NS_TIME,3),
             round(cupti_api_duration/NS_TIME,3),
             round(gpu_activity_time["kernelOps"]/NS_TIME,3),
             round(gpu_activity_time["memOps"]/NS_TIME,3),
             round(percgpu_activity,3)]
    
    return data

def remove_files(curr_dir):
    nsysfile = os.path.join(curr_dir,"gpu_estimation.nsys-rep")
    sqlitefile = os.path.join(curr_dir,"gpu_estimation.sqlite")
    subprocess.run(["rm",nsysfile], capture_output=True, text=True)
    subprocess.run(["rm",sqlitefile], capture_output=True, text=True)


def actual_main(args):
    result = subprocess.run(["which","nsys"], capture_output=True, text=True)
    if not result.stdout:
        print("Please make sure the command nsys is included in your path")
        print("You can try: export PATH=[path/to/bin]:$PATH")
        print("You can verify using:","\nwhich nsys","\nnsys --version")
        sys.exit(1)

    curr_dir = subprocess.run(["pwd"], capture_output=True, text=True).stdout.strip()
    nsys_output = subprocess.run(["nsys","profile","--trace=cuda,osrt","--cpuctxsw=none","--sample=none","--force-overwrite=true","--stats=true","--output=gpu_estimation","python", args.path_to_file], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True)
    if nsys_output.stderr:
        print("An error ocurred during the analysis")
        print("Please make sure that your training is executing on GPU")
        print("Error:",nsys_output.stderr)
        # remove generated files
        remove_files(curr_dir)
        sys.exit(1)

    db_path = os.path.join(curr_dir,"gpu_estimation.sqlite")
    summary = sql_command_execution(db_path)
    headers = ["Estimate Profiling time","CUDA API Time","Kernel Ops Time","Memory Ops time","GPU Perc"]
    format_row = "{:^25}" * len((headers))
    print(format_row.format(*headers))
    print(format_row.format(*summary))
    # remove generated files
    remove_files(curr_dir)

def main(args):
    actual_main(args)
