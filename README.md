# AI File Renamer

This application uses OpenAI's GPT to intelligently rename files based on configured rules and patterns.

<strong>CAUTION: This program was generated with "vibe coding". It may have mistakes and may cause harm to your computer. I am not a Python programmer, so I am not qualified to say this is a trustworthy program. Use at your own risk! </strong>

## Features

- Configure multiple folders for file monitoring
- Custom AI prompts for each folder with support for creation date variables
- File pattern filtering
- Automatic file watching or manual triggering
- Preserves original file timestamps
- YAML-based configuration
- Can run as a macOS LaunchAgent service

## Setup

1. Copy and modify the `config.yaml` file with your settings:
   - Configure the folders you want to monitor
   - Set custom prompts for each folder (with optional date variables)
   - Specify file patterns to filter (optional)
   - Enable/disable watching for each folder

Example configuration:
```yaml
folders:
  - path: "/path/to/your/folder1"
    prompt: "Analyze this file and suggest a clear, descriptive filename that follows these rules: 
            1. Start with the creation date: {created_date}
            2. Use lowercase
            3. Replace spaces with hyphens
            4. Keep the original file extension
            5. Make it descriptive of the content"
    file_pattern: "*.{jpg,png,pdf}"
    watch: true

  - path: "/path/to/your/folder2"
    prompt: "Rename this file to follow this format: {created_year}/{created_month}/{created_day}-title-of-content"
    file_pattern: "*.txt"
    watch: false
    debug: false
```

### Available Date Variables

You can use the following variables in your prompts to include file creation dates:
- `{created_date}` - Date in YYYY-MM-DD format
- `{created_datetime}` - Date and time in YYYY-MM-DD_HH-MM-SS format
- `{created_iso}` - Full ISO format timestamp
- `{created_year}` - Year (YYYY)
- `{created_month}` - Month (MM)
- `{created_day}` - Day (DD)
- `{created_time}` - Time in HH-MM-SS format

### Notes

- The script uses OpenAI's GPT-3.5-turbo model for generating new filenames
- File pattern matching supports glob patterns (e.g., `*.{jpg,png,pdf}`)
- Each folder can have its own naming convention through custom prompts
- Watching is optional per folder and can be enabled/disabled in the config
- Creation date variables can be used in prompts to include timestamps in filenames 

## Usage

### First-Time Setup
1. Use the `cd` command to change directory to this program's location.

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Install dependencies in the virtual environment:
   ```bash
   # Activate the virtual environment first
   source venv/bin/activate
   
   # Then install dependencies
   pip install -r requirements.txt
   ```

4. Configure the environment file. 
   ```bash
   cp .env.example .env
   ```
   - Add an OpenAI key to the new `.env` file.


### Running the Script
1. Activate the virtual environment (if not already activated):
   ```bash
   source venv/bin/activate
   ```

2. Run the script:
   ```bash
   # Option 1: If virtual environment is already activated
   python file_renamer.py
   
   # Option 2: Activate and run in one command
   source venv/bin/activate && python file_renamer.py
   ```

3. The script will:
   - Process any existing files in the configured folders
   - Start watching folders that have `watch: true`
   - Automatically rename new files as they are added to watched folders
   - Preserve original file timestamps

4. To stop the script, press Ctrl+C



## Running as a macOS Service

You can run the file renamer as a background service that starts automatically at login:

1. Use the `cd` command to change directory to this program's location.

2. Modify the `com.ai-file-renamer.plist` file
   - Replace `/Users/jn/Apps/ai-file-renamer` with the correct path for your file system

3. Create the LaunchAgent configuration:
   ```bash
   cat > ~/Library/LaunchAgents/com.ai-file-renamer.plist ./com.ai-file-renamer.plist
   ```

4. Create the logs directory:
   ```bash
   mkdir -p ~/Library/Logs/ai-file-renamer
   ```

### Service Management

To control the service, use these commands:

- **Start the service:**
  ```bash
  launchctl load ~/Library/LaunchAgents/com.ai-file-renamer.plist
  ```

- **Stop the service:**
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.ai-file-renamer.plist
  ```

- **Restart the service** (after configuration changes):
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.ai-file-renamer.plist && launchctl load ~/Library/LaunchAgents/com.ai-file-renamer.plist
  ```

- **Check service status:**
  ```bash
  launchctl list | grep com.ai-file-renamer
  ```

- **View logs:**
  ```bash
  # View output log
  tail -f ~/Library/Logs/ai-file-renamer/output.log
  
  # View error log
  tail -f ~/Library/Logs/ai-file-renamer/error.log
  ``` 