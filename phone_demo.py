import argparse
from transformations.mp_hand_detection import ImageTransform, MediaPipeHandDetectionTransform


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live processing of video from phone')

    parser.add_argument('url', type=str, help='URL of video stream')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument(
        '--app_type',
        choices=[
            'sync',                   # Synchronous mode
            'async_queue',            # Asynchronous mode
            'async_sharedmem',        # Asynchronous mode with shared memory
        ],
        default='sync',
        help='Type of app to run'
    )

    args = parser.parse_args()

    if args.app_type == 'sync':
        from stream_processing.sync_stream_processing_app import SynchronousStreamProcessingApp
        app = SynchronousStreamProcessingApp(
            args.url,
            MediaPipeHandDetectionTransform(),
            debug=args.debug
        )
    elif args.app_type == 'async_queue':
        from stream_processing.stream_processing_app import StreamProcessingApp
        app = StreamProcessingApp(
            args.url,
            MediaPipeHandDetectionTransform(),
            debug=args.debug
        )
    elif args.app_type == 'async_sharedmem':
        from stream_processing.sharedmem_stream_processing_app import SharedmemStreamProcessingApp
        app = SharedmemStreamProcessingApp(
            args.url,
            MediaPipeHandDetectionTransform(),
            debug=args.debug
        )
    app.run()

