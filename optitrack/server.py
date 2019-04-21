from optitrack.NatNetClient import NatNetClient
from optitrack.quarternion import Quaternion
import logging

logging.basicConfig(filename="optitrack_log", level=logging.DEBUG)
logging.info("Starting Logger")

data = [0.0 for _ in range(6)]

debug =False

# This is a callback function that gets connected to the NatNet client and called once per mocap frame.
def receiveNewFrame( frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
    # print( "Received frame", frameNumber )
    pass

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame( id, position, rotation ):
    quart = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
    euler = quart.euler
    if debug:
        print( "Received frame for rigid body", id )
        print( "Position: ", position )
        logging.info( "Position: {}".format(position))
        print( "Rotation: ", euler)
        logging.info( "Rotation: {}".format(euler))
    data[0] = position[0]
    data[1] = position[1]
    data[2] = position[2]
    data[3] = euler[0]
    data[4] = euler[1]
    data[5] = euler[2]

def StartServer():
    # This will create a new NatNet client
    streamingClient = NatNetClient()

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streamingClient.newFrameListener = receiveNewFrame
    streamingClient.rigidBodyListener = receiveRigidBodyFrame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    streamingClient.run()

if __name__ == "__main__":
    debug = True
    StartServer()
