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
        self.analyzer =  None # TODO ConversationAnalyzer(anthropic_api_key)

    async def handle_websocket(self, websocket: websockets.WebSocketServerProtocol):
        self.websocket = websocket
        logging.info(f"Client connected from {websocket.remote_address}")

        def on_transcription_data(transcript: aai.RealtimeTranscript):
            if not transcript.text:
                return
            
            # Only analyze final transcripts
            if isinstance(transcript, aai.RealtimeFinalTranscript):
                # TODO uncomment analysis references after testing transcription
                # analysis = self.analyzer.analyze_transcript(transcript.text)
                
                response = {
                    "type": "transcript",
                    "text": transcript.text,
                    "is_final": True,
                    "playbook_updates": [] # analysis.get("updates", [])
                }
            else:
                response = {
                    "type": "transcript",
                    "text": transcript.text,
                    "is_final": False
                }
            
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps(response)),
                asyncio.get_event_loop()
            )

        def on_transcription_error(error: aai.RealtimeError):
            logging.error(f"Transcription error: {error}")
            asyncio.run_coroutine_threadsafe(
                websocket.send(json.dumps({"type": "error", "error": str(error)})),
                asyncio.get_event_loop()
            )

        def transcription_thread():
            self.transcriber = aai.RealtimeTranscriber(
                on_data=on_transcription_data,
                on_error=on_transcription_error,
                sample_rate=44100
            )
            
            self.transcriber.connect()
            
            while self.is_running:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    self.transcriber.stream(audio_data)
            
            self.transcriber.close()

        try:
            self.is_running = True
            thread = threading.Thread(target=transcription_thread)
            thread.start()

            async for message in websocket:
                # debugging purposes to see the form of the message
                logging.info(f"type of message is ${type(message)} \nand message is: ${message}")
                self.audio_queue.put(message['media']['payload'])

        except websockets.exceptions.ConnectionClosed:
            logging.info("Client disconnected")
        finally:
            self.is_running = False
            if self.transcriber:
                self.transcriber.close()

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