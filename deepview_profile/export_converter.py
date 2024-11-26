import json


def convert(message):
    new_message = {}
    with open("message.json", "w") as fp:
        json.dump(message, fp, indent=4)

    new_message["ddp"] = {}
    new_message["message_type"] = message["message_type"]
    new_message["project_root"] = message["project_root"]
    new_message["project_entry_point"] = message["project_entry_point"]

    new_message["hardware_info"] = {
        "hostname": message["hardware_info"]["hostname"],
        "os": message["hardware_info"]["os"],
        "gpus": message["hardware_info"]["gpus"],
    }

    new_message["throughput"] = {
        "samples_per_second": message["throughput"]["samples_per_second"],
        "predicted_max_samples_per_second": message["throughput"][
            "predicted_max_samples_per_second"
        ],
        "run_time_ms": (
            [
                message["throughput"]["run_time_ms"]["slope"],
                message["throughput"]["run_time_ms"]["bias"],
            ]
            if "run_time_ms" in message["throughput"]
            else [0, 0]
        ),
        "peak_usage_bytes": (
            [
                message["throughput"]["peak_usage_bytes"]["slope"],
                message["throughput"]["peak_usage_bytes"]["bias"],
            ]
            if "peak_usage_bytes" in message["throughput"]
            else [0, 0]
        ),
        "batch_size_context": None,
        "can_manipulate_batch_size": False,
    }

    new_message["utilization"] = message["utilization"]

    def fix(a):
        for d in ["cpu", "gpu"]:
            for s in ["Forward", "Backward"]:
                if f"{d}_{s.lower()}" in a:
                    a[f"{d}{s}"] = a[f"{d}_{s.lower()}"]
                    del a[f"{d}_{s.lower()}"]
                else:
                    a[f"{d}{s}"] = 0

                if f"{d}_{s.lower()}_span" in a:
                    a[f"{d}{s}Span"] = a[f"{d}_{s.lower()}_span"]
                    del a[f"{d}_{s.lower()}_span"]
                else:
                    a[f"{d}{s}Span"] = 0

        if "children" not in a:
            a["children"] = []
            return

        if a:
            for c in a["children"]:
                fix(c)

    (
        fix(new_message["utilization"]["rootNode"])
        if new_message["utilization"].get("rootNode", None)
        else None
    )
    try:
        new_message["utilization"]["tensor_core_usage"] = message["utilization"][
            "tensor_utilization"
        ]
    except:
        new_message["utilization"]["tensor_core_usage"] = 0

    new_message["habitat"] = {
        "predictions": [
            (
                [prediction["device_name"], prediction["runtime_ms"]]
                if prediction["device_name"] != "unavailable"
                else ["default_device", 0]
            )
            for prediction in message["habitat"]["predictions"]
        ]
    }

    new_message["breakdown"] = {
        "peak_usage_bytes": int(message["breakdown"]["peak_usage_bytes"]),
        "memory_capacity_bytes": int(message["breakdown"]["memory_capacity_bytes"]),
        "iteration_run_time_ms": message["breakdown"]["iteration_run_time_ms"],
        # TODO change these hardcoded numbers
        "batch_size": 48,
        "num_nodes_operation_tree": len(message["breakdown"]["operation_tree"]),
        "num_nodes_weight_tree": 0,
        "operation_tree": [
            {
                "name": op["name"],
                "num_children": op["num_children"] if "num_children" in op else 0,
                "forward_ms": op["operation"]["forward_ms"],
                "backward_ms": op["operation"]["backward_ms"],
                "size_bytes": (
                    int(op["operation"]["size_bytes"])
                    if "size_bytes" in op["operation"]
                    else 0
                ),
                "file_refs": (
                    [
                        {
                            "path": "/".join(ctx["context"]["file_path"]["components"]),
                            "line_no": ctx["context"]["line_number"],
                            "run_time_ms": ctx["run_time_ms"],
                            "size_bytes": (
                                int(ctx["size_bytes"]) if "size_bytes" in ctx else 0
                            ),
                        }
                        for ctx in op["operation"]["context_info_map"]
                    ]
                    if "context_info_map" in op["operation"]
                    else list()
                ),
            }
            for op in message["breakdown"]["operation_tree"]
        ],
    }

    def fix_components(m):
        for c in m["components"]:
            if "consumption_joules" not in c:
                c["consumption"] = 0
            else:
                c["consumption"] = c["consumption_joules"]
                del c["consumption_joules"]
            c["type"] = c["component_type"]
            if c["type"] == "ENERGY_NVIDIA":
                c["type"] = "ENERGY_GPU"
            del c["component_type"]

    new_message["energy"] = {
        "current": {
            "total_consumption": message["energy"]["total_consumption"],
            "components": message["energy"]["components"],
            "batch_size": 48,
        },
        "past_measurements": message["energy"].get("past_measurements", None),
    }

    fix_components(new_message["energy"]["current"])
    if new_message["energy"].get("past_measurements", None):
        for m in new_message["energy"]["past_measurements"]:
            fix_components(m)

    return new_message
