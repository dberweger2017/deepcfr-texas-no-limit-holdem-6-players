# telegram_notifier.py
import requests
import traceback
import time
import os
from datetime import datetime
from dotenv import load_dotenv

class TelegramNotifier:
    def __init__(self, token=None, chat_id=None):
        """
        Initialize the Telegram notifier.
        If token and chat_id are not provided, they will be read from environment variables.
        """
        # Load environment variables
        load_dotenv()
        
        # Use provided values or read from environment
        self.token = token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        
        # Validate credentials were loaded
        if not self.token:
            raise ValueError("Telegram bot token not provided and not found in environment variables")
        if not self.chat_id:
            raise ValueError("Telegram chat ID not provided and not found in environment variables")
            
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.message_count = 0
        self.last_message_time = 0
        self.rate_limit = 5  # seconds between messages to avoid flooding
        self.training_start_time = datetime.now()
        
        # Test connection on startup
        self.send_message("🤖 POKER AI MONITORING ACTIVATED")
    
    def send_message(self, message, force=False):
        """Send a message to the Telegram chat."""
        # Rate limiting to avoid Telegram API restrictions
        current_time = time.time()
        if not force and current_time - self.last_message_time < self.rate_limit:
            # Wait to respect rate limits
            time.sleep(self.rate_limit - (current_time - self.last_message_time))
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            response.raise_for_status()
            self.message_count += 1
            self.last_message_time = time.time()
            return True
        except Exception as e:
            print(f"Failed to send Telegram notification: {e}")
            return False
    
    def alert_illegal_action(self, iteration, player_id, action, state):
        """Send alert about an illegal action."""
        message = f"⚠️ <b>ILLEGAL ACTION DETECTED</b> ⚠️\n\n"
        message += f"Iteration: {iteration}\n"
        message += f"Player: {player_id}\n"
        message += f"Action: {action}\n"
        message += f"Current Player Stake: {state.players_state[player_id].stake}\n"
        message += f"Current Player Bet: {state.players_state[player_id].bet_chips}\n"
        message += f"Pot Size: {state.pot}\n"
        message += f"Min Bet: {state.min_bet}\n"
        
        return self.send_message(message)
    
    def alert_state_error(self, iteration, status, state_before):
        """Send alert about a state error."""
        message = f"🚨 <b>STATE ERROR DETECTED</b> 🚨\n\n"
        message += f"Iteration: {iteration}\n"
        message += f"Status: {status}\n"
        message += f"Stage: {state_before.stage}\n"
        message += f"Pot: {state_before.pot}\n"
        message += f"Current Player: {state_before.current_player}\n"
        message += f"Legal Actions: {state_before.legal_actions}\n"
        
        return self.send_message(message)
    
    def alert_zero_reward_games(self, iteration, zero_rewards, total_games):
        """Send alert about games with zero rewards."""
        message = f"⚠️ <b>ZERO REWARD GAMES DETECTED</b> ⚠️\n\n"
        message += f"Iteration: {iteration}\n"
        message += f"Zero Reward Games: {zero_rewards}/{total_games}\n"
        message += f"Percentage: {zero_rewards/total_games*100:.1f}%\n"
        
        return self.send_message(message)
    
    def send_training_progress(self, iteration, profit_vs_models, profit_vs_random):
        """Send periodic training summary."""
        runtime = datetime.now() - self.training_start_time
        hours = runtime.total_seconds() // 3600
        minutes = (runtime.total_seconds() % 3600) // 60
        
        message = f"📊 <b>TRAINING PROGRESS</b>\n\n"
        message += f"Iteration: {iteration}\n"
        message += f"Runtime: {int(hours)}h {int(minutes)}m\n"
        message += f"Profit vs Models: {profit_vs_models:.2f}\n"
        message += f"Profit vs Random: {profit_vs_random:.2f}\n"
        
        return self.send_message(message)