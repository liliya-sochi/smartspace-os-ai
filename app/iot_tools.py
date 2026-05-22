import asyncio
from app.schema import SmartHomeState

async def set_climate_sync(zone: str, target_temp: float, home_state: SmartHomeState) -> str:
    """Изменяет уставку температуры климатического контроля в указанной зоне."""
    await asyncio.sleep(0.1)  # Симуляция задержки отправки команды в шину KNX/Modbus
    
    if zone.lower() in ["гостиная", "living_room", "living room"]:
        home_state.living_room_temp = target_temp
        return f"Успешно: В зоне {zone} установлена температура {target_temp}°C"
    
    return f"Ошибка: Зона {zone} не найдена в системе"

async def set_lighting_scene_sync(room: str, scene: str, home_state: SmartHomeState) -> str:
    """Устанавливает сценарий освещения ('ON', 'OFF', 'COZY', 'CINEMA') в выбранной комнате."""
    await asyncio.sleep(0.1)
    
    if room.lower() in ["кухня", "kitchen"]:
        home_state.kitchen_light = scene.upper()
        return f"Успешно: В комнате {room} активирован сценарий {scene}"
    
    return f"Ошибка: Комната {room} не найдена в системе"
