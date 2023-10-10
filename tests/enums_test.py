import pytest
import pandas as pd
from src.mesoscaler.enums import ERA5, Dimensions, URMA, Z, X, Y, LAT, LON, LVL, TIME, T
from src.mesoscaler.enum_table import TableEnum, auto_field


def test_coordinate_axes() -> None:
    assert LVL.axis == (Z,)
    assert TIME.axis == (T,)
    assert LAT.axis == (Y, X)
    assert LON.axis == (Y, X)


def test_main():
    with pytest.raises(ValueError):
        ERA5("nope")

    assert ERA5.Z == "geopotential"
    assert ERA5("z") == "geopotential"
    assert ERA5(["z"]) == ["geopotential"]
    assert ERA5(["z", "q"]) == ["geopotential", "specific_humidity"]
    assert ERA5.to_list() == [
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
    ]
    assert URMA[["TCC"]] == ["total_cloud_cover"]
    assert isinstance(Dimensions.to_frame(), pd.DataFrame)
    assert LAT.axis == (Y, X)
    assert ERA5("z") is ERA5.Z and ERA5.Z is ERA5("geopotential") and ERA5.Z == "geopotential"
    assert ERA5(("z", "t")) == ERA5(iter("zt"))
    assert ERA5("z") == ERA5.Z
    assert set(ERA5.difference(iter("tuv"))) == set(ERA5).difference(ERA5(iter("tuv")))


class MyEnum(str, TableEnum, my_class_metadata="hello"):
    A = auto_field("a", aliases=["alpha"], hello="world")
    B = auto_field("b", aliases=["beta"])
    C = auto_field("c", aliases=["beta"])
    D = auto_field("d", aliases=[4, 5, 6])


def test_my_enum() -> None:
    assert MyEnum.A == "a"
    assert MyEnum[["A", "B"]] == [MyEnum.A, MyEnum.B]
    assert MyEnum["A"] == "a" == MyEnum.A == MyEnum("alpha")
    assert MyEnum.A.metadata == {"hello": "world"}


def test_my_enum_metadata() -> None:
    assert set(MyEnum.__metadata__.keys()) == {"name", "member_metadata", "data", "class_metadata"}  # type: ignore
    assert MyEnum.__metadata__["name"] == "MyEnum"


def test_my_enum_class_metadata() -> None:
    class_meta = MyEnum.__metadata__["class_metadata"]
    assert class_meta is MyEnum.metadata


def test_member_metadata() -> None:
    member_meta = MyEnum.A.metadata
    assert member_meta is MyEnum.__metadata__["member_metadata"]["A"]

    mm = MyEnum._member_metadata
    assert MyEnum.__metadata__["member_metadata"] is mm
    assert mm == {"A": {"hello": "world"}, "B": {}, "C": {}, "D": {}}

    with pytest.raises(TypeError):
        # new metadata can't be added to the individual members
        mm["A"] = {"a": 1}  # type: ignore
    # with pytest.raises(TypeError):
    #     # and the data cannot be changed
    #     mm["A"]["hello"] = "mars"  # type: ignore

    assert mm["A"] == {"hello": "world"} == MyEnum.A.metadata
    assert MyEnum.B.metadata == {}
