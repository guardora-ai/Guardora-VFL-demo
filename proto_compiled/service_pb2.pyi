from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Request(_message.Message):
    __slots__ = ("party_name", "processor", "pub_key", "left_space", "type", "model_name", "split_index", "look_up_id", "lr", "factor", "grad", "hess", "array_float", "train_hash", "valid_hash", "instance_space", "index_space", "index")
    class serialized_encrypted_number(_message.Message):
        __slots__ = ("v", "e")
        V_FIELD_NUMBER: _ClassVar[int]
        E_FIELD_NUMBER: _ClassVar[int]
        v: str
        e: int
        def __init__(self, v: _Optional[str] = ..., e: _Optional[int] = ...) -> None: ...
    PARTY_NAME_FIELD_NUMBER: _ClassVar[int]
    PROCESSOR_FIELD_NUMBER: _ClassVar[int]
    PUB_KEY_FIELD_NUMBER: _ClassVar[int]
    LEFT_SPACE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    SPLIT_INDEX_FIELD_NUMBER: _ClassVar[int]
    LOOK_UP_ID_FIELD_NUMBER: _ClassVar[int]
    LR_FIELD_NUMBER: _ClassVar[int]
    FACTOR_FIELD_NUMBER: _ClassVar[int]
    GRAD_FIELD_NUMBER: _ClassVar[int]
    HESS_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FLOAT_FIELD_NUMBER: _ClassVar[int]
    TRAIN_HASH_FIELD_NUMBER: _ClassVar[int]
    VALID_HASH_FIELD_NUMBER: _ClassVar[int]
    INSTANCE_SPACE_FIELD_NUMBER: _ClassVar[int]
    INDEX_SPACE_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    party_name: str
    processor: str
    pub_key: str
    left_space: str
    type: str
    model_name: str
    split_index: int
    look_up_id: int
    lr: float
    factor: float
    grad: _containers.RepeatedCompositeFieldContainer[Request.serialized_encrypted_number]
    hess: _containers.RepeatedCompositeFieldContainer[Request.serialized_encrypted_number]
    array_float: _containers.RepeatedScalarFieldContainer[float]
    train_hash: _containers.RepeatedScalarFieldContainer[str]
    valid_hash: _containers.RepeatedScalarFieldContainer[str]
    instance_space: _containers.RepeatedScalarFieldContainer[str]
    index_space: _containers.RepeatedScalarFieldContainer[str]
    index: int
    def __init__(self, party_name: _Optional[str] = ..., processor: _Optional[str] = ..., pub_key: _Optional[str] = ..., left_space: _Optional[str] = ..., type: _Optional[str] = ..., model_name: _Optional[str] = ..., split_index: _Optional[int] = ..., look_up_id: _Optional[int] = ..., lr: _Optional[float] = ..., factor: _Optional[float] = ..., grad: _Optional[_Iterable[_Union[Request.serialized_encrypted_number, _Mapping]]] = ..., hess: _Optional[_Iterable[_Union[Request.serialized_encrypted_number, _Mapping]]] = ..., array_float: _Optional[_Iterable[float]] = ..., train_hash: _Optional[_Iterable[str]] = ..., valid_hash: _Optional[_Iterable[str]] = ..., instance_space: _Optional[_Iterable[str]] = ..., index_space: _Optional[_Iterable[str]] = ..., index: _Optional[int] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ("party_name", "status", "grad", "hess", "train_hash", "valid_hash", "array_float", "left_space", "right_space", "index")
    class serialized_encrypted_number(_message.Message):
        __slots__ = ("v", "e")
        V_FIELD_NUMBER: _ClassVar[int]
        E_FIELD_NUMBER: _ClassVar[int]
        v: str
        e: int
        def __init__(self, v: _Optional[str] = ..., e: _Optional[int] = ...) -> None: ...
    class serialized_encrypted_numbers(_message.Message):
        __slots__ = ("arr_serialized_encrypted_number",)
        ARR_SERIALIZED_ENCRYPTED_NUMBER_FIELD_NUMBER: _ClassVar[int]
        arr_serialized_encrypted_number: _containers.RepeatedCompositeFieldContainer[Response.serialized_encrypted_number]
        def __init__(self, arr_serialized_encrypted_number: _Optional[_Iterable[_Union[Response.serialized_encrypted_number, _Mapping]]] = ...) -> None: ...
    PARTY_NAME_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    GRAD_FIELD_NUMBER: _ClassVar[int]
    HESS_FIELD_NUMBER: _ClassVar[int]
    TRAIN_HASH_FIELD_NUMBER: _ClassVar[int]
    VALID_HASH_FIELD_NUMBER: _ClassVar[int]
    ARRAY_FLOAT_FIELD_NUMBER: _ClassVar[int]
    LEFT_SPACE_FIELD_NUMBER: _ClassVar[int]
    RIGHT_SPACE_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    party_name: str
    status: str
    grad: _containers.RepeatedCompositeFieldContainer[Response.serialized_encrypted_number]
    hess: _containers.RepeatedCompositeFieldContainer[Response.serialized_encrypted_number]
    train_hash: _containers.RepeatedScalarFieldContainer[str]
    valid_hash: _containers.RepeatedScalarFieldContainer[str]
    array_float: _containers.RepeatedScalarFieldContainer[float]
    left_space: _containers.RepeatedScalarFieldContainer[str]
    right_space: _containers.RepeatedScalarFieldContainer[str]
    index: int
    def __init__(self, party_name: _Optional[str] = ..., status: _Optional[str] = ..., grad: _Optional[_Iterable[_Union[Response.serialized_encrypted_number, _Mapping]]] = ..., hess: _Optional[_Iterable[_Union[Response.serialized_encrypted_number, _Mapping]]] = ..., train_hash: _Optional[_Iterable[str]] = ..., valid_hash: _Optional[_Iterable[str]] = ..., array_float: _Optional[_Iterable[float]] = ..., left_space: _Optional[_Iterable[str]] = ..., right_space: _Optional[_Iterable[str]] = ..., index: _Optional[int] = ...) -> None: ...
