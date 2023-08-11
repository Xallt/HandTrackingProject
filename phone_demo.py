import argparse
from transformations.mp_hand_detection import ImageTransform, MediaPipeHandDetectionTransform
from transformations.toy_transformations import NegativeTransformation


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
    parser.add_argument(
        '--transformation',
        choices=[
            'negative',               # Negative transformation
            'mp_hand_detection',      # MediaPipe hand detection
        ],
        default='mp_hand_detection',
        help='Transformation to apply to video stream'
    )
    parser.add_argument(
        '--async_transform',
        action='store_true',
        help='Use asynchronous transform'
    )

    args = parser.parse_args()
    if args.transformation == 'negative':
        transform = NegativeTransformation()
    elif args.transformation == 'mp_hand_detection':
        transform = MediaPipeHandDetectionTransform(enable_mouse=False)

    if args.app_type == 'sync':
        from stream_processing.sync_stream_processing_app import SynchronousStreamProcessingApp
        app = SynchronousStreamProcessingApp(
            args.url,
            transform,
            debug=args.debug
        )
    elif args.app_type == 'async_queue':
        from stream_processing.queue_stream_processing_app import QueueStreamProcessingApp
        app = QueueStreamProcessingApp(
            args.url,
            transform,
            debug=args.debug
        )
    elif args.app_type == 'async_sharedmem':
        from stream_processing.sharedmem_stream_processing_app import SharedmemStreamProcessingApp
        app = SharedmemStreamProcessingApp(
            args.url,
            transform,
            debug=args.debug,
            async_transform=args.async_transform
        )
    app.run()

