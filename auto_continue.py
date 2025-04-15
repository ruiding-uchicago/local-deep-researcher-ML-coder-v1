import time
import datetime
import os
import glob # Added for log file checking
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

# --- Configuration ---
LANGSMITH_STUDIO_URL = "https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
LOGS_DIRECTORY = "./logs" # Added: Path to your LangGraph logs

# --- IMPORTANT: Update this selector if the LangSmith UI changes ---
CONTINUE_BUTTON_XPATH = '//button[contains(@class, "MuiButton-root") and contains(@class, "MuiButton-variantSolid") and contains(., "Continue")]' # Using XPath as it's better for text contains

# --- Selectors to detect run state (NEEDS ADJUSTMENT BASED ON ACTUAL UI) ---
STATUS_INDICATOR_SELECTOR = "div[data-testid='run-status-badge']" # Example: Find the status badge
COMPLETED_STATUS_TEXT = ["completed", "failed", "error"] # Status texts indicating run end
# LAST_UPDATED_SELECTOR = "span[data-testid='run-last-updated']" # Example: Find timestamp span (Less reliable than logs)

MONITORING_INTERVAL_SECONDS = 300  # Check every 5 minutes
STALL_THRESHOLD_SECONDS = 60     # Consider stalled if no update (UI or logs) for 1 minute AND continue is visible
INITIAL_WAIT_SECONDS = 30          # Added: Wait after page load before first check

# --- Helper Functions ---
def get_current_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_latest_log_modification_time(log_dir):
    """Checks the logs directory and returns the latest modification timestamp."""
    latest_mod_time = 0
    try:
        # Use recursive glob to find all files if logs are nested (adjust pattern if needed)
        # list_of_files = glob.glob(os.path.join(log_dir, '**/*'), recursive=True) 
        list_of_files = glob.glob(os.path.join(log_dir, '*')) # Non-recursive for flat structure
        if not list_of_files:
            return None # No log files found

        # Filter out directories, only check files
        files_only = [f for f in list_of_files if os.path.isfile(f)]
        if not files_only:
             return None # No files found
        
        latest_file = max(files_only, key=os.path.getmtime)
        latest_mod_time = os.path.getmtime(latest_file)
        return datetime.datetime.fromtimestamp(latest_mod_time)
    except FileNotFoundError:
        print(f"[{get_current_time_str()}] Log directory not found: {log_dir}")
        return None
    except Exception as e:
        print(f"[{get_current_time_str()}] Error checking log directory {log_dir}: {e}")
        return None

def find_element_safe(driver, by, value, timeout=5):
    try:
        # Use visibility_of_element_located for more reliability
        return WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((by, value))
        )
    except TimeoutException:
        return None
    except Exception as e:
        print(f"[{get_current_time_str()}] Error finding element ({by}={value}): {e}")
        return None

def click_element_safe(element):
    # Element should already be checked for visibility by find_element_safe
    if element and element.is_enabled():
        try:
            # Add a slight pause before click, sometimes helps with dynamic UIs
            time.sleep(0.5)
            element.click()
            print(f"[{get_current_time_str()}] Successfully clicked element.")
            return True
        except StaleElementReferenceException:
            print(f"[{get_current_time_str()}] Click failed: Stale element reference. Will retry find.")
            return False # Indicate failure to allow retry
        except Exception as e:
            print(f"[{get_current_time_str()}] Click failed: {e}")
            return False
    else:
        # Condition check might be redundant if find_element_safe uses visibility
        print(f"[{get_current_time_str()}] Click failed: Element not found, not visible, or not enabled.")
        return False

# --- Main Script ---
print(f"[{get_current_time_str()}] Initializing WebDriver...")
try:
    service = ChromeService(executable_path=ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Enable for headless run
    driver = webdriver.Chrome(service=service, options=options)
    print(f"[{get_current_time_str()}] WebDriver initialized.")
except Exception as e:
    print(f"[{get_current_time_str()}] Failed to initialize WebDriver: {e}")
    exit(1)

print(f"[{get_current_time_str()}] Navigating to LangSmith Studio: {LANGSMITH_STUDIO_URL}")
driver.get(LANGSMITH_STUDIO_URL)

print(f"[{get_current_time_str()}] Please manually log in to LangSmith if required within the automated browser.")
print(f"[{get_current_time_str()}] Then, navigate to the specific run/thread you want to monitor.")
print(f"[{get_current_time_str()}] Starting monitoring loop in {INITIAL_WAIT_SECONDS} seconds...")
time.sleep(INITIAL_WAIT_SECONDS) # Removed input, added wait

print(f"[{get_current_time_str()}] Monitoring started (Interval: {MONITORING_INTERVAL_SECONDS}s)...")

last_update_time = datetime.datetime.now() # Initialize with current time
last_known_status = ""
last_log_check_time = datetime.datetime.fromtimestamp(0) # Track last log mod time checked

try:
    while True:
        current_time = datetime.datetime.now()
        print(f"\n[{get_current_time_str()}] --- Checking Status ---")
        
        activity_detected_this_cycle = False

        # 1. Check Log Files for recent modification
        latest_log_time = get_latest_log_modification_time(LOGS_DIRECTORY)
        if latest_log_time:
            print(f"[{get_current_time_str()}] Latest log modification: {latest_log_time.strftime('%Y-%m-%d %H:%M:%S')}")
            if latest_log_time > last_log_check_time and latest_log_time > last_update_time:
                 print(f"[{get_current_time_str()}] Log activity detected since last check.")
                 last_update_time = latest_log_time # Update based on log activity
                 activity_detected_this_cycle = True
                 last_log_check_time = latest_log_time # Update the time we last saw a log change
        else:
             print(f"[{get_current_time_str()}] No log files found or error checking logs.")


        # 2. Check UI Status
        status_element = find_element_safe(driver, By.CSS_SELECTOR, STATUS_INDICATOR_SELECTOR)
        current_status = ""
        if status_element:
            try:
                current_status = status_element.text.lower().strip()
                print(f"[{get_current_time_str()}] Current UI run status detected: '{current_status}'")
                if current_status != last_known_status:
                    print(f"[{get_current_time_str()}] UI Status changed.")
                    # Only update if UI change is more recent than last log update
                    if current_time > last_update_time:
                         last_update_time = current_time # Status change counts as an update
                    activity_detected_this_cycle = True
                    last_known_status = current_status
                    
                if any(term in current_status for term in COMPLETED_STATUS_TEXT):
                    print(f"[{get_current_time_str()}] Run has completed or failed ('{current_status}'). Stopping monitor.")
                    break
            except Exception as e:
                 print(f"[{get_current_time_str()}] Error reading status element: {e}")
                 # If we can't read status, assume it might be running to prevent premature clicks
                 activity_detected_this_cycle = True 
        else:
             print(f"[{get_current_time_str()}] UI Status indicator element not found (Selector: {STATUS_INDICATOR_SELECTOR}).")
             # If no status, rely on logs / continue button presence. Maybe assume activity?
             # Setting activity_detected prevents clicking if UI status is broken but logs are active
             if latest_log_time and latest_log_time > last_update_time :
                 pass # Log activity already updated last_update_time
             else:
                 # If no log activity either, it *might* be stalled, but harder to tell
                 print(f"[{get_current_time_str()}] Cannot determine UI status reliably.")


        # 3. Check for Continue Button
        continue_button = None
        try:
            # Use the more reliable XPath
            continue_button = find_element_safe(driver, By.XPATH, CONTINUE_BUTTON_XPATH)
            if continue_button:
                 print(f"[{get_current_time_str()}] 'Continue' button IS visible.")
            else:
                 print(f"[{get_current_time_str()}] 'Continue' button IS NOT visible.")
                 # If continue button disappears, implies activity or completion
                 if not activity_detected_this_cycle and not any(term in last_known_status for term in COMPLETED_STATUS_TEXT):
                     print(f"[{get_current_time_str()}] Continue button gone, assuming activity resumed.")
                     last_update_time = current_time
                     activity_detected_this_cycle = True

        except Exception as e:
             print(f"[{get_current_time_str()}] Error checking for continue button: {e}")
             continue_button = None # Assume not found on error


        # --- Stall Detection Logic ---
        is_stalled = False
        if continue_button: # Only consider stall if button is visible
            time_since_last_update = (current_time - last_update_time).total_seconds()
            print(f"[{get_current_time_str()}] Time since last detected activity (UI or Logs): {time_since_last_update:.0f}s")

            if time_since_last_update > STALL_THRESHOLD_SECONDS:
                 if not any(term in last_known_status for term in COMPLETED_STATUS_TEXT): # Double check it's not completed
                    print(f"[{get_current_time_str()}] STALL DETECTED: 'Continue' button visible and no activity for > {STALL_THRESHOLD_SECONDS}s.")
                    is_stalled = True
                 else:
                      print(f"[{get_current_time_str()}] Threshold exceeded, but run status ('{last_known_status}') indicates completion/failure. Not clicking.")
            else:
                 print(f"[{get_current_time_str()}] 'Continue' button visible, but not considered stalled yet (Threshold: {STALL_THRESHOLD_SECONDS}s).")
        # No need for 'else' here, if continue_button not found, is_stalled remains False


        # --- Click Action ---
        if is_stalled and continue_button:
            print(f"[{get_current_time_str()}] Attempting to click 'Continue' button...")
            clicked = click_element_safe(continue_button)
            if clicked:
                # Assume clicking resets the state, update timestamp immediately
                last_update_time = datetime.datetime.now()
                last_known_status = "resumed_by_script" # Indicate we acted
                last_log_check_time = last_update_time # Reset log check time too
                print(f"[{get_current_time_str()}] Click successful. Short pause before next cycle...")
                time.sleep(15) # Short pause after clicking
            else:
                print(f"[{get_current_time_str()}] Click failed. Will retry finding element in next cycle.")
                # Don't update last_update_time if click failed
        
        # Update last time checked even if no activity, for stall calculation
        # No, last_update_time should only reflect actual activity
        # If no activity this cycle, time_since_last_update will just grow

        # --- Wait for next interval ---
        print(f"[{get_current_time_str()}] Waiting for {MONITORING_INTERVAL_SECONDS}s...")
        time.sleep(MONITORING_INTERVAL_SECONDS)

except KeyboardInterrupt:
    print(f"\n[{get_current_time_str()}] Monitoring interrupted by user.")
except Exception as e:
    print(f"\n[{get_current_time_str()}] An unexpected error occurred during monitoring: {e}")
finally:
    print(f"[{get_current_time_str()}] Shutting down WebDriver...")
    if 'driver' in locals() and driver:
        driver.quit()
    print(f"[{get_current_time_str()}] Monitor finished.")