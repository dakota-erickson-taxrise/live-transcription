import asyncio
import websockets
import assemblyai as aai
import json
from queue import Queue
import threading
from typing import Optional, Dict
from anthropic import Anthropic
import time
import os
import logging


logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO
)

class PlaybookItem:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.completed = False
        self.completed_text = ""
        self.timestamp = None

class ConversationAnalyzer:
    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.conversation_history = []
        self.playbook_items = [
            PlaybookItem("introduction", "Agent introduces themselves and their role"),
            PlaybookItem("situation_inquiry", "Agent asks about the customer's situation"),
            PlaybookItem("problem_identification", "Agent identifies the core problem"),
            PlaybookItem("solution_proposal", "Agent proposes potential solutions"),
            PlaybookItem("next_steps", "Agent outlines next steps or action items")
        ]
    
    def analyze_transcript(self, text: str) -> Dict:
        # Append new text to conversation history
        self.conversation_history.append(text)
        
        # Create prompt for Claude to analyze the conversation
        prompt = f"""Given the following conversation transcript and playbook items, identify which items have been completed.
        For each completed item, provide the relevant quote that demonstrates completion.
        
        Playbook items:
        {json.dumps([{'name': item.name, 'description': item.description} for item in self.playbook_items], indent=2)}
        
        Conversation transcript:
        {' '.join(self.conversation_history)}
        
        Respond in JSON format with completed items and their supporting quotes."""
        
        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                system="You are analyzing a customer service conversation to identify completed playbook items. Respond only with JSON.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            analysis = json.loads(response.content[0].text)
            
            # Update playbook items based on analysis
            updates = []
            for item in self.playbook_items:
                if item.name in analysis and not item.completed:
                    item.completed = True
                    item.completed_text = analysis[item.name]
                    item.timestamp = time.time()
                    updates.append({
                        "item": item.name,
                        "completed": True,
                        "text": item.completed_text,
                        "timestamp": item.timestamp
                    })
            
            return {"updates": updates}
            
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return {"error": str(e)}

class TranscriptionWebSocket:
    def __init__(self, assembly_api_key: str, anthropic_api_key: str):
        self.api_key = assembly_api_key
        aai.settings.api_key = self.api_key
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.transcriber = None
        self.audio_queue = Queue()
        self.is_running = False
        self.loop = None
        self.analyzer = None

    async def handle_websocket(self, websocket: websockets.WebSocketServerProtocol):
        self.websocket = websocket
        self.loop = asyncio.get_running_loop()
        logging.info(f"Client connected from {websocket.remote_address}")

        def on_transcription_data(transcript: aai.RealtimeTranscript):
            logging.info("Transcript received:")
            logging.info(f"  Type: {type(transcript)}")
            logging.info(f"  Content: {transcript.__dict__}")
            
            if isinstance(transcript, aai.RealtimeFinalTranscript):
                response = {
                    "type": "transcript",
                    "text": transcript.text if transcript.text else "",
                    "is_final": True,
                    "playbook_updates": [],
                    "words": [word.__dict__ for word in transcript.words] if hasattr(transcript, 'words') else []
                }
            else:
                response = {
                    "type": "transcript",
                    "text": transcript.text if transcript.text else "",
                    "is_final": False,
                    "words": [word.__dict__ for word in transcript.words] if hasattr(transcript, 'words') else []
                }

            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.send_message(response))
            )

        def on_transcription_error(error: aai.RealtimeError):
            logging.error(f"Transcription error: {error}")
            self.loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.send_message(
                    {"type": "error", "error": str(error)}
                ))
            )

        async def process_audio_queue():
            while self.is_running:
                if not self.audio_queue.empty():
                    message = self.audio_queue.get()
                    payload = message.get("payload", None)
                    
                    if payload is not None:
                        try:
                            # Log incoming payload type and length
                            logging.info(f"Received payload type: {type(payload)}")
                            logging.info(f"Payload length: {len(str(payload))}")
                            
                            # Handle different payload types
                            if isinstance(payload, str):
                                try:
                                    # Try to decode as base64
                                    audio_data = base64.b64decode(payload)
                                    logging.info(f"Decoded base64 data, length: {len(audio_data)} bytes")
                                except Exception as e:
                                    logging.error(f"Base64 decode failed: {e}")
                                    audio_data = payload.encode()
                            elif isinstance(payload, bytes):
                                audio_data = payload
                            else:
                                logging.error(f"Unexpected payload type: {type(payload)}")
                                continue

                            # Log audio data details
                            logging.info(f"Processing audio data: {len(audio_data)} bytes")
                            
                            # Stream to transcriber
                            self.transcriber.stream({"audio_data": audio_data})
                            
                        except Exception as e:
                            logging.error(f"Error processing audio data: {e}")
                            logging.error(f"Payload that caused error: {payload[:100]}...")  # Log first 100 chars
                            
                await asyncio.sleep(0.01)

        def transcription_thread():
            try:
                logging.info("Initializing transcriber...")
                
                self.transcriber = aai.RealtimeTranscriber(
                    on_data=on_transcription_data,
                    on_error=on_transcription_error,
                    sample_rate=44100,  # Using standard audio sample rate
                )

                logging.info("Connecting transcriber...")
                self.transcriber.connect()
                logging.info("Transcriber connected successfully")
                
                while self.is_running:
                    time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Error in transcription thread: {str(e)}")
                logging.error(f"Error type: {type(e)}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
            finally:
                logging.info("Closing transcriber connection")
                if self.transcriber:
                    self.transcriber.close()

        try:
            self.is_running = True
            thread = threading.Thread(target=transcription_thread)
            thread.start()

            audio_process_task = asyncio.create_task(process_audio_queue())

            async for message in websocket:
                try:
                    # Log raw message length
                    logging.info(f"Received websocket message length: {len(message)}")
                    
                    json_parsed_message = json.loads(message)
                    # Log message structure
                    logging.info(f"Message keys: {json_parsed_message.keys()}")
                    
                    self.audio_queue.put(json_parsed_message)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parse error: {e}")
                    logging.error(f"Failed message: {message[:100]}...")  # Log first 100 chars
                except Exception as e:
                    logging.error(f"Unexpected error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected")
        finally:
            self.is_running = False
            if self.transcriber:
                self.transcriber.close()
            audio_process_task.cancel()

    async def send_message(self, message):
        if self.websocket and self.websocket.open:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                logging.error(f"Error sending message: {e}")
async def start_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    assembly_api_key: str = None,
    anthropic_api_key: str = None
):
    
    # Exit early if we don't get API keys
    if assembly_api_key is None:
        logging.error("Missing AssemblyAI API key")
        return None
    if anthropic_api_key is None:
        logging.error("Missing AnthropicAI API key")
        return None
    
    transcription_ws = TranscriptionWebSocket(assembly_api_key, anthropic_api_key)
    
    async with websockets.serve(
        transcription_ws.handle_websocket,
        host,
        port
    ):
        logging.info(f"WebSocket server started on ws://{host}:{port}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    ASSEMBLY_API_KEY = os.environ.get("ASSEMBLY_AI_KEY")
    ANTHROPIC_API_KEY = "your_anthropic_api_key"
    
    logging.info("Running websocket...")

    asyncio.run(start_server(
        assembly_api_key=ASSEMBLY_API_KEY,
        anthropic_api_key=ANTHROPIC_API_KEY
    ))