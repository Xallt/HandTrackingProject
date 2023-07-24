import argparse
from transformations.mp_hand_detection import ImageTransform, MediaPipeHandDetectionTransform
from stream_processing.stream_processing_app import StreamProcessingApp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live processing of video from phone')

    parser.add_argument('url', type=str, help='URL of video stream')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    app = StreamProcessingApp(
        args.url,
        MediaPipeHandDetectionTransform(),
        debug=args.debug
    )
    app.run()

