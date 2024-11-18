import mavsdk
import asyncio


async def main():

    drone = mavsdk.System() ## MAVSDK object that represents drone

    await drone.connect(system_address="udp://:14540") ## default system address of SITL

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- Connected to drone!")
            break

    # arm the drone

    print("Arming...")
    await drone.action.arm()

    # finally, takeoff! 

    print("Taking off!")
    await drone.action.takeoff()

    await asyncio.sleep(10) # wait a while
    
    print("Landing!")
    await drone.action.land() # finally, land the drone




if __name__ == "__main__":
    asyncio.run(main())


