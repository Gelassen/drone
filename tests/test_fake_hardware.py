import pytest
from tests.mock.fake_hardware import FakeHardware


@pytest.mark.asyncio
async def test_connect_sets_connected():
    hw = FakeHardware()
    assert not hw.is_connected()

    await hw.connect()
    assert hw.is_connected()

@pytest.mark.asyncio
async def test_can_arm_returns_bool():
    hw = FakeHardware(can_arm=True)
    result = await hw.can_arm()
    assert isinstance(result, bool)

@pytest.mark.asyncio
async def test_arm_and_takeoff_success():
    hw = FakeHardware(can_arm=True)

    await hw.connect()
    await hw.arm_and_takeoff(2.5)

    assert ("arm_and_takeoff", 2.5) in hw.calls

@pytest.mark.asyncio
async def test_arm_and_takeoff_fails_if_cannot_arm():
    hw = FakeHardware(can_arm=False)

    await hw.connect()

    with pytest.raises(RuntimeError):
        await hw.arm_and_takeoff(1.5)

@pytest.mark.asyncio
async def test_arm_and_takeoff_success():
    hw = FakeHardware(can_arm=True)

    await hw.connect()
    await hw.arm_and_takeoff(2.5)

    assert ("arm_and_takeoff", 2.5) in hw.calls

@pytest.mark.asyncio
async def test_arm_and_takeoff_success():
    hw = FakeHardware(can_arm=True)

    await hw.connect()
    await hw.arm_and_takeoff(2.5)

    assert ("arm_and_takeoff", 2.5) in hw.calls

@pytest.mark.asyncio
async def test_land_always_resets_armed():
    hw = FakeHardware(can_arm=True)

    await hw.arm_and_takeoff(2.0)
    await hw.land()

    assert hw._armed is False
