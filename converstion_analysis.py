import os
import sys
import asyncio
import base64
import argparse
import threading
import queue
from azure.ai.voicelive.models import ServerEventType
from typing import Union, Optional, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
import logging

# Audio file support
import wave

# # Environment variable loading
# try:
#     from dotenv import load_dotenv
#     load_dotenv()
# except ImportError:
#     print("Note: python-dotenv not installed. Using existing environment variables.")

# Azure VoiceLive SDK imports
from azure.core.credentials import AzureKeyCredential, TokenCredential
# from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

from azure.ai.voicelive.aio import connect

if TYPE_CHECKING:
    from azure.ai.voicelive.aio import VoiceLiveConnection

from azure.ai.voicelive.models import (
    RequestSession,
    ServerVad,
    AzureStandardVoice,
    Modality,
    AudioFormat,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioFileProcessor:
    """
    Handles audio file reading for the voice assistant.
    Reads from a WAV/PCM audio file and sends to VoiceLive API.
    """

    def __init__(self, connection, audio_file_path: str):
        self.connection = connection
        self.audio_file_path = audio_file_path

        # Audio configuration - PCM16, 24kHz, mono
        self.chunk_size = 1024

        # Audio queues and threading
        self.audio_send_queue: "queue.Queue[str]" = queue.Queue()  # base64 audio to send
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.send_thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(f"AudioFileProcessor initialized for file: {audio_file_path}")

    async def start_file_playback(self):
        """Start reading and sending audio from file."""
        self.loop = asyncio.get_event_loop()

        # Start file send thread
        self.send_thread = threading.Thread(target=self._read_audio_file_thread)
        self.send_thread.daemon = True
        self.send_thread.start()

        logger.info("Started audio file processing")

    def _read_audio_file_thread(self):
        """Read audio file and queue it for sending."""
        try:
            # Try to open as WAV file first
            try:
                with wave.open(self.audio_file_path, 'rb') as wav_file:
                    # Read WAV file
                    while True:
                        audio_data = wav_file.readframes(self.chunk_size)
                        if not audio_data:
                            break

                        if self.loop:
                            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                            self.audio_send_queue.put(audio_base64)
                            # Small delay to avoid overwhelming the queue
                            import time
                            time.sleep(0.01)

            except wave.Error:
                # Fall back to raw audio file (assume PCM16, 24kHz, mono)
                logger.info("Not a WAV file, treating as raw PCM audio")
                with open(self.audio_file_path, 'rb') as f:
                    while True:
                        audio_data = f.read(self.chunk_size * 2)  # 16-bit = 2 bytes per sample
                        if not audio_data:
                            break

                        if self.loop:
                            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                            self.audio_send_queue.put(audio_base64)
                            import time
                            time.sleep(0.01)

            # Start the send thread after reading is queued
            self.send_thread = threading.Thread(target=self._send_audio_thread)
            self.send_thread.daemon = True
            self.send_thread.start()

            logger.info("Finished reading audio file")

        except FileNotFoundError:
            logger.error(f"Audio file not found: {self.audio_file_path}")
        except Exception as e:
            logger.error(f"Error reading audio file: {e}")

    def _send_audio_thread(self):
        """Send queued audio data to VoiceLive."""
        while True:
            try:
                # Get audio data from queue (blocking with timeout)
                audio_base64 = self.audio_send_queue.get(timeout=0.5)

                if audio_base64 and self.loop:
                    future = asyncio.run_coroutine_threadsafe(
                        self.connection.input_audio_buffer.append(audio=audio_base64), self.loop
                    )

            except queue.Empty:
                # Check if queue is empty and file reading is done
                if self.audio_send_queue.empty():
                    break
                continue
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                break

    async def cleanup(self):
        """Clean up audio resources."""
        self.executor.shutdown(wait=True)
        logger.info("Audio processor cleaned up")


class AudioFileAssistant:
    """Voice assistant that reads from audio file and generates text responses."""

    def __init__(
        self,
        endpoint: str,
        credential: Union[AzureKeyCredential, TokenCredential],
        model: str,
        voice: str,
        instructions: str,
        audio_file_path: str,
    ):

        self.endpoint = endpoint
        self.credential = credential
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self.audio_file_path = audio_file_path
        self.connection: Optional["VoiceLiveConnection"] = None
        self.audio_processor: Optional[AudioFileProcessor] = None
        self.session_ready = False
        self.response_text = ""

    async def start(self):
        """Start the voice assistant session."""
        try:
            logger.info(f"Connecting to VoiceLive API with model {self.model}")

            async with connect(
                endpoint=self.endpoint,
                credential=self.credential,
                model=self.model,
                connection_options={
                    "max_msg_size": 10 * 1024 * 1024,
                    "heartbeat": 20,
                    "timeout": 20,
                },
            ) as connection:
                conn = connection
                self.connection = conn

                # Initialize audio processor for file reading
                ap = AudioFileProcessor(conn, self.audio_file_path)
                self.audio_processor = ap

                # Configure session
                await self._setup_session()

                # Start audio file processing (no microphone needed)
                await ap.start_file_playback()

                logger.info("Processing audio file...")
                print("\n" + "=" * 60)
                print("🎵 PROCESSING AUDIO FILE")
                print(f"File: {self.audio_file_path}")
                print("=" * 60 + "\n")

                # Process events
                await self._process_events()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")

        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

        # Cleanup
        if self.audio_processor:
            await self.audio_processor.cleanup()

        # Print final response
        if self.response_text:
            print("\n" + "=" * 60)
            print("📝 ASSISTANT RESPONSE:")
            print("=" * 60)
            print(self.response_text)
            print("=" * 60 + "\n")

    async def _setup_session(self):
        """Configure the VoiceLive session."""
        logger.info("Setting up session...")

        voice_config: Union[AzureStandardVoice, str]
        if self.voice.startswith("en-US-") or self.voice.startswith("en-CA-") or "-" in self.voice:
            voice_config = AzureStandardVoice(name=self.voice, type="azure-standard")
        else:
            voice_config = self.voice

        turn_detection_config = ServerVad(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500)

        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=self.instructions,
            voice=voice_config,
            input_audio_format=AudioFormat.PCM16,
            output_audio_format=AudioFormat.PCM16,
            turn_detection=turn_detection_config,
        )

        conn = self.connection
        assert conn is not None, "Connection must be established before setting up session"
        await conn.session.update(session=session_config)

        logger.info("Session configuration sent")

    async def _process_events(self):
        """Process events from the VoiceLive connection."""
        try:
            conn = self.connection
            assert conn is not None, "Connection must be established before processing events"
            async for event in conn:
                await self._handle_event(event)

        except KeyboardInterrupt:
            logger.info("Event processing interrupted")
        except Exception as e:
            logger.error(f"Error processing events: {e}")
            raise

    async def _handle_event(self, event):
        """Handle different types of events from VoiceLive."""
        logger.debug(f"Received event: {event.type}")
        conn = self.connection
        assert conn is not None, "Connection must be established"

        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info(f"Session ready: {event.session.id}")
            self.session_ready = True

        elif event.type == ServerEventType.RESPONSE_CREATED:
            logger.info("🤖 Assistant response created")

        elif event.type == ServerEventType.RESPONSE_TEXT_DELTA:
            # Capture text responses
            logger.debug("Received text delta")
            if hasattr(event, 'delta'):
                self.response_text += event.delta
                print(event.delta, end='', flush=True)

        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            # Log audio response (not playing since no audio output)
            logger.debug("Received audio delta")

        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            logger.info("🤖 Assistant finished speaking")

        elif event.type == ServerEventType.RESPONSE_DONE:
            logger.info("✅ Response complete")

        elif event.type == ServerEventType.ERROR:
            logger.error(f"❌ VoiceLive error: {event.error.message}")
            print(f"Error: {event.error.message}")

        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug(f"Conversation item created: {event.item.id}")

        else:
            logger.debug(f"Unhandled event type: {event.type}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Audio File Voice Assistant using Azure VoiceLive SDK",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--audio-file",
        help="Path to audio file (WAV or PCM format)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--api-key",
        help="Azure VoiceLive API key. If not provided, will use AZURE_VOICELIVE_API_KEY environment variable.",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_API_KEY"),
    )

    parser.add_argument(
        "--endpoint",
        help="Azure VoiceLive endpoint",
        type=str,
        default=os.environ.get("AZURE_VOICELIVE_ENDPOINT", "wss://api.voicelive.com/v1"),
    )

    parser.add_argument(
        "--model",
        help="VoiceLive model to use",
        type=str,
        default=os.environ.get("VOICELIVE_MODEL", "gpt-4o-realtime-preview"),
    )

    parser.add_argument(
        "--voice",
        help="Voice to use for the assistant",
        type=str,
        default=os.environ.get("VOICELIVE_VOICE", "en-US-AvaNeural"),
    )

    parser.add_argument(
        "--instructions",
        help="System instructions for the AI assistant",
        type=str,
        default=os.environ.get(
            "VOICELIVE_INSTRUCTIONS",
            "You are a helpful AI assistant. Respond naturally and conversationally. "
            "Keep your responses concise but engaging.",
        ),
    )

    parser.add_argument("--verbose", help="Enable verbose logging", action="store_true")

    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.api_key:
        print("❌ Error: No API key provided")
        print("Please provide an API key using --api-key or set AZURE_VOICELIVE_API_KEY environment variable")
        sys.exit(1)

    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    try:
        credential = AzureKeyCredential(args.api_key)
        logger.info("Using API key credential")

        assistant = AudioFileAssistant(
            endpoint=args.endpoint,
            credential=credential,
            model=args.model,
            voice=args.voice,
            instructions=args.instructions,
            audio_file_path=args.audio_file,
        )

        await assistant.start()

    except KeyboardInterrupt:
        print("\n👋 Assistant shut down. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Check for required dependencies
    dependencies = {
        "azure.ai.voicelive": "Azure VoiceLive SDK",
        "azure.core": "Azure Core libraries",
    }

    missing_deps = []
    for dep, description in dependencies.items():
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing_deps.append(f"{dep} ({description})")

    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("Install with: pip install azure-ai-voicelive")
        sys.exit(1)

    print("🎙️  Audio File Voice Assistant with Azure VoiceLive SDK")
    print("=" * 50)

    asyncio.run(main())