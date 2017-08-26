import sys
import zmq
import msgpack
import time

sys.path.append("/Users/kai/Work/PupilLabs/git/pupil/pupil_src/shared_modules")
ctx = zmq.Context()

# establish connection
ip = 'localhost'
port = 50020
requester = ctx.socket(zmq.REQ)
requester.connect('tcp://%s:%s'%(ip, port))

# convenience functions
def send_recv_notification(n):
    topic = 'notify..%s'%n
    payload = msgpack.dumps(n)
    requester.send_string(topic, flags=zmq.SNDMORE)
    requester.send(payload)
    return requester.recv_string()

# start eye windows
file_path = '/Users/kai/Desktop/000/eye0.mp4'
cap_settings = ['File_Source', {'source_path': file_path, 'timed_playback': True}]
n = {'subject': 'eye_process.should_stop.0', 'eye_id': 0, 'args': {}}
print(send_recv_notification(n))
time.sleep(2)
n = {'subject': 'set_detection_mapping_mode', 'mode': '3d'}
print(send_recv_notification(n))
n = {'subject': 'eye_process.should_start.0', 'eye_id': 0, 'args': {}, 'overwrite_cap_settings': cap_settings}
print(send_recv_notification(n))
