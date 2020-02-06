# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: innpv.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='innpv.proto',
  package='innpv.protocol',
  syntax='proto3',
  serialized_pb=_b('\n\x0binnpv.proto\x12\x0einnpv.protocol\"\x9e\x01\n\nFromClient\x12\x17\n\x0fsequence_number\x18\x01 \x01(\r\x12\x37\n\ninitialize\x18\x02 \x01(\x0b\x32!.innpv.protocol.InitializeRequestH\x00\x12\x33\n\x08\x61nalysis\x18\x03 \x01(\x0b\x32\x1f.innpv.protocol.AnalysisRequestH\x00\x42\t\n\x07payload\"-\n\x11InitializeRequest\x12\x18\n\x10protocol_version\x18\x01 \x01(\r\"(\n\x0f\x41nalysisRequest\x12\x15\n\rmock_response\x18\x01 \x01(\x08\"\xe9\x02\n\nFromServer\x12\x17\n\x0fsequence_number\x18\x01 \x01(\r\x12.\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x1d.innpv.protocol.ProtocolErrorH\x00\x12\x38\n\ninitialize\x18\x03 \x01(\x0b\x32\".innpv.protocol.InitializeResponseH\x00\x12\x37\n\x0e\x61nalysis_error\x18\x05 \x01(\x0b\x32\x1d.innpv.protocol.AnalysisErrorH\x00\x12\x38\n\nthroughput\x18\x06 \x01(\x0b\x32\".innpv.protocol.ThroughputResponseH\x00\x12\x36\n\tbreakdown\x18\x08 \x01(\x0b\x32!.innpv.protocol.BreakdownResponseH\x00\x42\t\n\x07payloadJ\x04\x08\x04\x10\x05J\x04\x08\x07\x10\x08R\x0cmemory_usageR\x08run_time\"\\\n\x12InitializeResponse\x12\x1b\n\x13server_project_root\x18\x01 \x01(\t\x12)\n\x0b\x65ntry_point\x18\x02 \x01(\x0b\x32\x14.innpv.protocol.Path\"&\n\rAnalysisError\x12\x15\n\rerror_message\x18\x01 \x01(\t\"Z\n\x12ThroughputResponse\x12\x1a\n\x12samples_per_second\x18\x01 \x01(\x02\x12(\n predicted_max_samples_per_second\x18\x02 \x01(\x02\"\xd6\x01\n\x11\x42reakdownResponse\x12\x18\n\x10peak_usage_bytes\x18\x01 \x01(\x04\x12\x1d\n\x15memory_capacity_bytes\x18\x02 \x01(\x04\x12\x1d\n\x15iteration_run_time_ms\x18\x03 \x01(\x02\x12\x35\n\x0eoperation_tree\x18\x04 \x03(\x0b\x32\x1d.innpv.protocol.BreakdownNode\x12\x32\n\x0bweight_tree\x18\x05 \x03(\x0b\x32\x1d.innpv.protocol.BreakdownNode\"\xca\x01\n\rProtocolError\x12;\n\nerror_code\x18\x01 \x01(\x0e\x32\'.innpv.protocol.ProtocolError.ErrorCode\"|\n\tErrorCode\x12\x0b\n\x07UNKNOWN\x10\x00\x12 \n\x1cUNSUPPORTED_PROTOCOL_VERSION\x10\x01\x12\x1c\n\x18UNINITIALIZED_CONNECTION\x10\x02\x12\"\n\x1e\x41LREADY_INITIALIZED_CONNECTION\x10\x03\"\x1a\n\x04Path\x12\x12\n\ncomponents\x18\x01 \x03(\t\"M\n\rFileReference\x12\'\n\tfile_path\x18\x01 \x01(\x0b\x32\x14.innpv.protocol.Path\x12\x13\n\x0bline_number\x18\x02 \x01(\r\"\xce\x01\n\rBreakdownNode\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x14\n\x0cnum_children\x18\x02 \x01(\r\x12/\n\x08\x63ontexts\x18\x03 \x03(\x0b\x32\x1d.innpv.protocol.FileReference\x12\x32\n\toperation\x18\x04 \x01(\x0b\x32\x1d.innpv.protocol.OperationDataH\x00\x12,\n\x06weight\x18\x05 \x01(\x0b\x32\x1a.innpv.protocol.WeightDataH\x00\x42\x06\n\x04\x64\x61ta\"{\n\x0b\x43ontextInfo\x12.\n\x07\x63ontext\x18\x01 \x01(\x0b\x32\x1d.innpv.protocol.FileReference\x12\x13\n\x0brun_time_ms\x18\x02 \x01(\x02\x12\x12\n\nsize_bytes\x18\x03 \x01(\x04\x12\x13\n\x0binvocations\x18\x04 \x01(\r\"\x83\x01\n\rOperationData\x12\x12\n\nforward_ms\x18\x01 \x01(\x02\x12\x13\n\x0b\x62\x61\x63kward_ms\x18\x02 \x01(\x02\x12\x12\n\nsize_bytes\x18\x03 \x01(\x04\x12\x35\n\x10\x63ontext_info_map\x18\x04 \x03(\x0b\x32\x1b.innpv.protocol.ContextInfo\"9\n\nWeightData\x12\x12\n\nsize_bytes\x18\x01 \x01(\x04\x12\x17\n\x0fgrad_size_bytes\x18\x02 \x01(\x04\"\x1b\n\x13MemoryUsageResponseJ\x04\x08\x01\x10\x65\"\x17\n\x0fRunTimeResponseJ\x04\x08\x01\x10\x65\"\x17\n\x0f\x41\x63tivationEntryJ\x04\x08\x01\x10\x65\"\x13\n\x0bWeightEntryJ\x04\x08\x01\x10\x65\"\x14\n\x0cRunTimeEntryJ\x04\x08\x01\x10\x65\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_PROTOCOLERROR_ERRORCODE = _descriptor.EnumDescriptor(
  name='ErrorCode',
  full_name='innpv.protocol.ProtocolError.ErrorCode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNSUPPORTED_PROTOCOL_VERSION', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNINITIALIZED_CONNECTION', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALREADY_INITIALIZED_CONNECTION', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1167,
  serialized_end=1291,
)
_sym_db.RegisterEnumDescriptor(_PROTOCOLERROR_ERRORCODE)


_FROMCLIENT = _descriptor.Descriptor(
  name='FromClient',
  full_name='innpv.protocol.FromClient',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sequence_number', full_name='innpv.protocol.FromClient.sequence_number', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='initialize', full_name='innpv.protocol.FromClient.initialize', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='analysis', full_name='innpv.protocol.FromClient.analysis', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='payload', full_name='innpv.protocol.FromClient.payload',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=32,
  serialized_end=190,
)


_INITIALIZEREQUEST = _descriptor.Descriptor(
  name='InitializeRequest',
  full_name='innpv.protocol.InitializeRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='protocol_version', full_name='innpv.protocol.InitializeRequest.protocol_version', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=192,
  serialized_end=237,
)


_ANALYSISREQUEST = _descriptor.Descriptor(
  name='AnalysisRequest',
  full_name='innpv.protocol.AnalysisRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='mock_response', full_name='innpv.protocol.AnalysisRequest.mock_response', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=239,
  serialized_end=279,
)


_FROMSERVER = _descriptor.Descriptor(
  name='FromServer',
  full_name='innpv.protocol.FromServer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sequence_number', full_name='innpv.protocol.FromServer.sequence_number', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='error', full_name='innpv.protocol.FromServer.error', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='initialize', full_name='innpv.protocol.FromServer.initialize', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='analysis_error', full_name='innpv.protocol.FromServer.analysis_error', index=3,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='throughput', full_name='innpv.protocol.FromServer.throughput', index=4,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='breakdown', full_name='innpv.protocol.FromServer.breakdown', index=5,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='payload', full_name='innpv.protocol.FromServer.payload',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=282,
  serialized_end=643,
)


_INITIALIZERESPONSE = _descriptor.Descriptor(
  name='InitializeResponse',
  full_name='innpv.protocol.InitializeResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='server_project_root', full_name='innpv.protocol.InitializeResponse.server_project_root', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='entry_point', full_name='innpv.protocol.InitializeResponse.entry_point', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=645,
  serialized_end=737,
)


_ANALYSISERROR = _descriptor.Descriptor(
  name='AnalysisError',
  full_name='innpv.protocol.AnalysisError',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='error_message', full_name='innpv.protocol.AnalysisError.error_message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=739,
  serialized_end=777,
)


_THROUGHPUTRESPONSE = _descriptor.Descriptor(
  name='ThroughputResponse',
  full_name='innpv.protocol.ThroughputResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='samples_per_second', full_name='innpv.protocol.ThroughputResponse.samples_per_second', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='predicted_max_samples_per_second', full_name='innpv.protocol.ThroughputResponse.predicted_max_samples_per_second', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=779,
  serialized_end=869,
)


_BREAKDOWNRESPONSE = _descriptor.Descriptor(
  name='BreakdownResponse',
  full_name='innpv.protocol.BreakdownResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='peak_usage_bytes', full_name='innpv.protocol.BreakdownResponse.peak_usage_bytes', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='memory_capacity_bytes', full_name='innpv.protocol.BreakdownResponse.memory_capacity_bytes', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='iteration_run_time_ms', full_name='innpv.protocol.BreakdownResponse.iteration_run_time_ms', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='operation_tree', full_name='innpv.protocol.BreakdownResponse.operation_tree', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight_tree', full_name='innpv.protocol.BreakdownResponse.weight_tree', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=872,
  serialized_end=1086,
)


_PROTOCOLERROR = _descriptor.Descriptor(
  name='ProtocolError',
  full_name='innpv.protocol.ProtocolError',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='error_code', full_name='innpv.protocol.ProtocolError.error_code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _PROTOCOLERROR_ERRORCODE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1089,
  serialized_end=1291,
)


_PATH = _descriptor.Descriptor(
  name='Path',
  full_name='innpv.protocol.Path',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='components', full_name='innpv.protocol.Path.components', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1293,
  serialized_end=1319,
)


_FILEREFERENCE = _descriptor.Descriptor(
  name='FileReference',
  full_name='innpv.protocol.FileReference',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='file_path', full_name='innpv.protocol.FileReference.file_path', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='line_number', full_name='innpv.protocol.FileReference.line_number', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1321,
  serialized_end=1398,
)


_BREAKDOWNNODE = _descriptor.Descriptor(
  name='BreakdownNode',
  full_name='innpv.protocol.BreakdownNode',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='innpv.protocol.BreakdownNode.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_children', full_name='innpv.protocol.BreakdownNode.num_children', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='contexts', full_name='innpv.protocol.BreakdownNode.contexts', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='operation', full_name='innpv.protocol.BreakdownNode.operation', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='weight', full_name='innpv.protocol.BreakdownNode.weight', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='data', full_name='innpv.protocol.BreakdownNode.data',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=1401,
  serialized_end=1607,
)


_CONTEXTINFO = _descriptor.Descriptor(
  name='ContextInfo',
  full_name='innpv.protocol.ContextInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='innpv.protocol.ContextInfo.context', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='run_time_ms', full_name='innpv.protocol.ContextInfo.run_time_ms', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='size_bytes', full_name='innpv.protocol.ContextInfo.size_bytes', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='invocations', full_name='innpv.protocol.ContextInfo.invocations', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1609,
  serialized_end=1732,
)


_OPERATIONDATA = _descriptor.Descriptor(
  name='OperationData',
  full_name='innpv.protocol.OperationData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='forward_ms', full_name='innpv.protocol.OperationData.forward_ms', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='backward_ms', full_name='innpv.protocol.OperationData.backward_ms', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='size_bytes', full_name='innpv.protocol.OperationData.size_bytes', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='context_info_map', full_name='innpv.protocol.OperationData.context_info_map', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1735,
  serialized_end=1866,
)


_WEIGHTDATA = _descriptor.Descriptor(
  name='WeightData',
  full_name='innpv.protocol.WeightData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='size_bytes', full_name='innpv.protocol.WeightData.size_bytes', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='grad_size_bytes', full_name='innpv.protocol.WeightData.grad_size_bytes', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1868,
  serialized_end=1925,
)


_MEMORYUSAGERESPONSE = _descriptor.Descriptor(
  name='MemoryUsageResponse',
  full_name='innpv.protocol.MemoryUsageResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1927,
  serialized_end=1954,
)


_RUNTIMERESPONSE = _descriptor.Descriptor(
  name='RunTimeResponse',
  full_name='innpv.protocol.RunTimeResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1956,
  serialized_end=1979,
)


_ACTIVATIONENTRY = _descriptor.Descriptor(
  name='ActivationEntry',
  full_name='innpv.protocol.ActivationEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1981,
  serialized_end=2004,
)


_WEIGHTENTRY = _descriptor.Descriptor(
  name='WeightEntry',
  full_name='innpv.protocol.WeightEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2006,
  serialized_end=2025,
)


_RUNTIMEENTRY = _descriptor.Descriptor(
  name='RunTimeEntry',
  full_name='innpv.protocol.RunTimeEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2027,
  serialized_end=2047,
)

_FROMCLIENT.fields_by_name['initialize'].message_type = _INITIALIZEREQUEST
_FROMCLIENT.fields_by_name['analysis'].message_type = _ANALYSISREQUEST
_FROMCLIENT.oneofs_by_name['payload'].fields.append(
  _FROMCLIENT.fields_by_name['initialize'])
_FROMCLIENT.fields_by_name['initialize'].containing_oneof = _FROMCLIENT.oneofs_by_name['payload']
_FROMCLIENT.oneofs_by_name['payload'].fields.append(
  _FROMCLIENT.fields_by_name['analysis'])
_FROMCLIENT.fields_by_name['analysis'].containing_oneof = _FROMCLIENT.oneofs_by_name['payload']
_FROMSERVER.fields_by_name['error'].message_type = _PROTOCOLERROR
_FROMSERVER.fields_by_name['initialize'].message_type = _INITIALIZERESPONSE
_FROMSERVER.fields_by_name['analysis_error'].message_type = _ANALYSISERROR
_FROMSERVER.fields_by_name['throughput'].message_type = _THROUGHPUTRESPONSE
_FROMSERVER.fields_by_name['breakdown'].message_type = _BREAKDOWNRESPONSE
_FROMSERVER.oneofs_by_name['payload'].fields.append(
  _FROMSERVER.fields_by_name['error'])
_FROMSERVER.fields_by_name['error'].containing_oneof = _FROMSERVER.oneofs_by_name['payload']
_FROMSERVER.oneofs_by_name['payload'].fields.append(
  _FROMSERVER.fields_by_name['initialize'])
_FROMSERVER.fields_by_name['initialize'].containing_oneof = _FROMSERVER.oneofs_by_name['payload']
_FROMSERVER.oneofs_by_name['payload'].fields.append(
  _FROMSERVER.fields_by_name['analysis_error'])
_FROMSERVER.fields_by_name['analysis_error'].containing_oneof = _FROMSERVER.oneofs_by_name['payload']
_FROMSERVER.oneofs_by_name['payload'].fields.append(
  _FROMSERVER.fields_by_name['throughput'])
_FROMSERVER.fields_by_name['throughput'].containing_oneof = _FROMSERVER.oneofs_by_name['payload']
_FROMSERVER.oneofs_by_name['payload'].fields.append(
  _FROMSERVER.fields_by_name['breakdown'])
_FROMSERVER.fields_by_name['breakdown'].containing_oneof = _FROMSERVER.oneofs_by_name['payload']
_INITIALIZERESPONSE.fields_by_name['entry_point'].message_type = _PATH
_BREAKDOWNRESPONSE.fields_by_name['operation_tree'].message_type = _BREAKDOWNNODE
_BREAKDOWNRESPONSE.fields_by_name['weight_tree'].message_type = _BREAKDOWNNODE
_PROTOCOLERROR.fields_by_name['error_code'].enum_type = _PROTOCOLERROR_ERRORCODE
_PROTOCOLERROR_ERRORCODE.containing_type = _PROTOCOLERROR
_FILEREFERENCE.fields_by_name['file_path'].message_type = _PATH
_BREAKDOWNNODE.fields_by_name['contexts'].message_type = _FILEREFERENCE
_BREAKDOWNNODE.fields_by_name['operation'].message_type = _OPERATIONDATA
_BREAKDOWNNODE.fields_by_name['weight'].message_type = _WEIGHTDATA
_BREAKDOWNNODE.oneofs_by_name['data'].fields.append(
  _BREAKDOWNNODE.fields_by_name['operation'])
_BREAKDOWNNODE.fields_by_name['operation'].containing_oneof = _BREAKDOWNNODE.oneofs_by_name['data']
_BREAKDOWNNODE.oneofs_by_name['data'].fields.append(
  _BREAKDOWNNODE.fields_by_name['weight'])
_BREAKDOWNNODE.fields_by_name['weight'].containing_oneof = _BREAKDOWNNODE.oneofs_by_name['data']
_CONTEXTINFO.fields_by_name['context'].message_type = _FILEREFERENCE
_OPERATIONDATA.fields_by_name['context_info_map'].message_type = _CONTEXTINFO
DESCRIPTOR.message_types_by_name['FromClient'] = _FROMCLIENT
DESCRIPTOR.message_types_by_name['InitializeRequest'] = _INITIALIZEREQUEST
DESCRIPTOR.message_types_by_name['AnalysisRequest'] = _ANALYSISREQUEST
DESCRIPTOR.message_types_by_name['FromServer'] = _FROMSERVER
DESCRIPTOR.message_types_by_name['InitializeResponse'] = _INITIALIZERESPONSE
DESCRIPTOR.message_types_by_name['AnalysisError'] = _ANALYSISERROR
DESCRIPTOR.message_types_by_name['ThroughputResponse'] = _THROUGHPUTRESPONSE
DESCRIPTOR.message_types_by_name['BreakdownResponse'] = _BREAKDOWNRESPONSE
DESCRIPTOR.message_types_by_name['ProtocolError'] = _PROTOCOLERROR
DESCRIPTOR.message_types_by_name['Path'] = _PATH
DESCRIPTOR.message_types_by_name['FileReference'] = _FILEREFERENCE
DESCRIPTOR.message_types_by_name['BreakdownNode'] = _BREAKDOWNNODE
DESCRIPTOR.message_types_by_name['ContextInfo'] = _CONTEXTINFO
DESCRIPTOR.message_types_by_name['OperationData'] = _OPERATIONDATA
DESCRIPTOR.message_types_by_name['WeightData'] = _WEIGHTDATA
DESCRIPTOR.message_types_by_name['MemoryUsageResponse'] = _MEMORYUSAGERESPONSE
DESCRIPTOR.message_types_by_name['RunTimeResponse'] = _RUNTIMERESPONSE
DESCRIPTOR.message_types_by_name['ActivationEntry'] = _ACTIVATIONENTRY
DESCRIPTOR.message_types_by_name['WeightEntry'] = _WEIGHTENTRY
DESCRIPTOR.message_types_by_name['RunTimeEntry'] = _RUNTIMEENTRY

FromClient = _reflection.GeneratedProtocolMessageType('FromClient', (_message.Message,), dict(
  DESCRIPTOR = _FROMCLIENT,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.FromClient)
  ))
_sym_db.RegisterMessage(FromClient)

InitializeRequest = _reflection.GeneratedProtocolMessageType('InitializeRequest', (_message.Message,), dict(
  DESCRIPTOR = _INITIALIZEREQUEST,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.InitializeRequest)
  ))
_sym_db.RegisterMessage(InitializeRequest)

AnalysisRequest = _reflection.GeneratedProtocolMessageType('AnalysisRequest', (_message.Message,), dict(
  DESCRIPTOR = _ANALYSISREQUEST,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.AnalysisRequest)
  ))
_sym_db.RegisterMessage(AnalysisRequest)

FromServer = _reflection.GeneratedProtocolMessageType('FromServer', (_message.Message,), dict(
  DESCRIPTOR = _FROMSERVER,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.FromServer)
  ))
_sym_db.RegisterMessage(FromServer)

InitializeResponse = _reflection.GeneratedProtocolMessageType('InitializeResponse', (_message.Message,), dict(
  DESCRIPTOR = _INITIALIZERESPONSE,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.InitializeResponse)
  ))
_sym_db.RegisterMessage(InitializeResponse)

AnalysisError = _reflection.GeneratedProtocolMessageType('AnalysisError', (_message.Message,), dict(
  DESCRIPTOR = _ANALYSISERROR,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.AnalysisError)
  ))
_sym_db.RegisterMessage(AnalysisError)

ThroughputResponse = _reflection.GeneratedProtocolMessageType('ThroughputResponse', (_message.Message,), dict(
  DESCRIPTOR = _THROUGHPUTRESPONSE,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.ThroughputResponse)
  ))
_sym_db.RegisterMessage(ThroughputResponse)

BreakdownResponse = _reflection.GeneratedProtocolMessageType('BreakdownResponse', (_message.Message,), dict(
  DESCRIPTOR = _BREAKDOWNRESPONSE,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.BreakdownResponse)
  ))
_sym_db.RegisterMessage(BreakdownResponse)

ProtocolError = _reflection.GeneratedProtocolMessageType('ProtocolError', (_message.Message,), dict(
  DESCRIPTOR = _PROTOCOLERROR,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.ProtocolError)
  ))
_sym_db.RegisterMessage(ProtocolError)

Path = _reflection.GeneratedProtocolMessageType('Path', (_message.Message,), dict(
  DESCRIPTOR = _PATH,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.Path)
  ))
_sym_db.RegisterMessage(Path)

FileReference = _reflection.GeneratedProtocolMessageType('FileReference', (_message.Message,), dict(
  DESCRIPTOR = _FILEREFERENCE,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.FileReference)
  ))
_sym_db.RegisterMessage(FileReference)

BreakdownNode = _reflection.GeneratedProtocolMessageType('BreakdownNode', (_message.Message,), dict(
  DESCRIPTOR = _BREAKDOWNNODE,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.BreakdownNode)
  ))
_sym_db.RegisterMessage(BreakdownNode)

ContextInfo = _reflection.GeneratedProtocolMessageType('ContextInfo', (_message.Message,), dict(
  DESCRIPTOR = _CONTEXTINFO,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.ContextInfo)
  ))
_sym_db.RegisterMessage(ContextInfo)

OperationData = _reflection.GeneratedProtocolMessageType('OperationData', (_message.Message,), dict(
  DESCRIPTOR = _OPERATIONDATA,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.OperationData)
  ))
_sym_db.RegisterMessage(OperationData)

WeightData = _reflection.GeneratedProtocolMessageType('WeightData', (_message.Message,), dict(
  DESCRIPTOR = _WEIGHTDATA,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.WeightData)
  ))
_sym_db.RegisterMessage(WeightData)

MemoryUsageResponse = _reflection.GeneratedProtocolMessageType('MemoryUsageResponse', (_message.Message,), dict(
  DESCRIPTOR = _MEMORYUSAGERESPONSE,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.MemoryUsageResponse)
  ))
_sym_db.RegisterMessage(MemoryUsageResponse)

RunTimeResponse = _reflection.GeneratedProtocolMessageType('RunTimeResponse', (_message.Message,), dict(
  DESCRIPTOR = _RUNTIMERESPONSE,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.RunTimeResponse)
  ))
_sym_db.RegisterMessage(RunTimeResponse)

ActivationEntry = _reflection.GeneratedProtocolMessageType('ActivationEntry', (_message.Message,), dict(
  DESCRIPTOR = _ACTIVATIONENTRY,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.ActivationEntry)
  ))
_sym_db.RegisterMessage(ActivationEntry)

WeightEntry = _reflection.GeneratedProtocolMessageType('WeightEntry', (_message.Message,), dict(
  DESCRIPTOR = _WEIGHTENTRY,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.WeightEntry)
  ))
_sym_db.RegisterMessage(WeightEntry)

RunTimeEntry = _reflection.GeneratedProtocolMessageType('RunTimeEntry', (_message.Message,), dict(
  DESCRIPTOR = _RUNTIMEENTRY,
  __module__ = 'innpv_pb2'
  # @@protoc_insertion_point(class_scope:innpv.protocol.RunTimeEntry)
  ))
_sym_db.RegisterMessage(RunTimeEntry)


# @@protoc_insertion_point(module_scope)
