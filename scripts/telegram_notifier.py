# telegram_notifier.py
import pokers as pkrs
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
    
    # In telegram_notifier.py, update the alert_state_error method:

    def alert_state_error(self, iteration, status, state_before, is_training_agent=False):
        """Send enhanced alert about a state error with detailed betting information."""
        message = f"🚨 <b>STATE ERROR DETECTED</b> 🚨\n\n"
        message += f"Iteration: {iteration}\n"
        message += f"Status: {status}\n"
        message += f"<b>{'TRAINING AGENT' if is_training_agent else 'OPPONENT MODEL'}</b>\n"
        message += f"Stage: {state_before.stage}\n"
        message += f"Pot: {state_before.pot}\n"
        message += f"Current Player: {state_before.current_player}\n"
        
        # Add player stake info
        player_state = state_before.players_state[state_before.current_player]
        player_stake = player_state.stake
        player_bet = player_state.bet_chips
        
        # Add detailed betting information
        message += f"Player Stake: {player_stake}\n"
        message += f"Player Current Bet: {player_bet}\n"
        message += f"Min Bet: {state_before.min_bet}\n"
        
        # Calculate acceptable bet ranges
        if status == pkrs.StateStatus.HighBet:
            message += f"\n<b>BET VALIDATION DETAILS:</b>\n"
            message += f"Maximum legal bet: {player_stake}\n"
            
            # For all-in situations
            if player_stake < state_before.min_bet:
                message += f"All-in required (stake < min bet)\n"
            else:
                message += f"Legal bet range: {state_before.min_bet} to {player_stake}\n"
            
            # Calculate half pot and pot raises
            half_pot = state_before.pot * 0.5
            full_pot = state_before.pot
            message += f"Half pot raise: {half_pot}\n"
            message += f"Full pot raise: {full_pot}\n"
            
            # Show if these raises would be legal
            message += f"Half pot raise legal: {'YES' if half_pot <= player_stake else 'NO'}\n"
            message += f"Full pot raise legal: {'YES' if full_pot <= player_stake else 'NO'}\n"
            
        message += f"Legal Actions: {state_before.legal_actions}\n"
        
        # Add action that caused the error if available
        if state_before.from_action:
            action = state_before.from_action.action
            message += f"\nPrevious action: {action.action}"
            if action.action == pkrs.ActionEnum.Raise:
                message += f" {action.amount}"
        
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

    def debug_bet_calculation(self, state, action_id, raise_amount, iteration):
        """Send detailed bet calculation information."""
        player_state = state.players_state[state.current_player]
        player_stake = player_state.stake
        player_bet = player_state.bet_chips
        
        message = f"🔍 <b>BET CALCULATION DEBUG</b>\n\n"
        message += f"Iteration: {iteration}\n"
        message += f"Player: {state.current_player}\n"
        message += f"Action ID: {action_id} ({'0.5x pot' if action_id == 2 else '1x pot'})\n\n"
        
        message += f"<b>STATE DETAILS:</b>\n"
        message += f"Stage: {state.stage}\n"
        message += f"Pot: {state.pot}\n"
        message += f"Player Stake: {player_stake}\n"
        message += f"Player Current Bet: {player_bet}\n"
        message += f"Min Bet: {state.min_bet}\n\n"
        
        message += f"<b>BET VALIDATION:</b>\n"
        message += f"Calculated raise: {raise_amount}\n"
        message += f"All-in required: {'YES' if player_stake < state.min_bet else 'NO'}\n"
        message += f"Half pot amount: {state.pot * 0.5}\n"
        message += f"Full pot amount: {state.pot}\n"
        message += f"Legal raise range: {state.min_bet} to {player_stake}\n"
        
        return self.send_message(message)