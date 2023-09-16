import argparse
import time
from lib.comms.orb_comms import ORBcomms
from lib.visualization.video import play_trip
from eric_VO import VisualOdometry


def test_serialization(oc: ORBcomms, keypoints1, descriptors1):
    data = oc._serialize_output(keypoints1, descriptors1)
    keypoints2, descriptors2  = oc.deserialize_input(data)

    assert len(keypoints1) == len(keypoints2)





def main():
    data_dir = 'KITTI_sequence_2'  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)
    oc = ORBcomms(mode="send", ip="127.0.0.1", port=7000)

    # play_trip(vo.images)  # Comment out to not play the trip
    start_time = time.time()
    for i in range(0,len(vo.images)):
        
        print(f"{i}/{len(vo.images)}")
        keypoints1, descriptors1 = vo.make_keypoints(vo.images[i], None)
        test_serialization(oc, keypoints1, descriptors1)
        oc.send_orb_data(keypoints1, descriptors1)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
    print(f"fps = {len(vo.images)/(end_time - start_time)}")


if __name__ == "__main__":
    main()
