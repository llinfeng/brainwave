# Brainwave: Real-Time Speech Recognition and Summarization Tool

## Table of Contents

1. [Introduction](#introduction)
2. [Deployment](#deployment)
3. [Code Structure & Architecture](#code-structure--architecture)
4. [Testing](#testing)

---

## Introduction

### Background

In the era of rapid information exchange, capturing and organizing ideas swiftly is paramount. **Brainwave** addresses this need by providing a robust speech recognition input method that allows users to effortlessly input their thoughts, regardless of their initial organization. Leveraging advanced technologies, Brainwave transforms potentially messy and unstructured verbal inputs into coherent and logical summaries, enhancing productivity and idea management.

### Goals

- **Efficient Speech Recognition:** Enable users to quickly input ideas through speech, reducing the friction of manual typing.
- **Organized Summarization:** Automatically process and summarize spoken input into structured and logical formats.
- **Multilingual Support:** Cater to a diverse user base by supporting multiple languages, ensuring accessibility and convenience.

### Technical Advantages

1. **Real-Time Processing:**
   - **Low Latency:** Processes audio streams in real-time, providing immediate transcription and summarization, which is essential for maintaining the flow of thoughts.
   - **Continuous Interaction:** Unlike traditional batch processing systems, Brainwave offers seamless real-time interaction, ensuring that users receive timely response on their inputs.

2. **Multilingual Proficiency:**
   - **Diverse Language Support:** Handles inputs in multiple languages without the need for separate processing pipelines, enhancing versatility and user accessibility.
   - **Automatic Language Detection:** Identifies the language of the input automatically, streamlining the user experience.

3. **Sophisticated Text Processing:**
   - **Error Correction:** Utilizes advanced algorithms to identify and correct errors inherent in speech recognition, ensuring accurate transcriptions.
   - **Readability Enhancement:** Improves punctuation and structure of the transcribed text, making summaries clear and professional.
   - **Intent Recognition:** Understands the context and intent behind the spoken words, enabling the generation of meaningful summaries.

---

## Deployment

Deploying **Brainwave** involves setting up a Python-based environment, installing the necessary dependencies, and launching the server to handle real-time speech recognition and summarization. Follow the steps below to get started:

### Prerequisites

- **Python 3.11+**: Ensure that Python 3.11 or higher is installed on your system.
- **uv**: A fast Python package manager and project manager. Install it from [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Setup Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/llinfeng/brainwave.git
   cd brainwave
   ```

2. **Install Dependencies and Create Virtual Environment**

   Using uv, install all dependencies and create a virtual environment automatically:

   ```bash
   uv sync
   ```

   This command will:
   - Create a virtual environment in `.venv/`
   - Install all dependencies from `pyproject.toml`
   - Set up the project for development

   **To reinitialize/refresh the virtual environment:**
   ```bash
   # Reinstall all packages (useful after environment issues or updates)
   uv sync --reinstall
   
   # Or completely remove and recreate the virtual environment
   rm -rf .venv
   uv sync
   ```

   **Virtual Environment Usage Options:**
   
   With uv, you have two ways to work with the virtual environment:
   
   - **Option 1 (Recommended): Use `uv run`** - No activation needed
     ```bash
     uv run python script.py
     uv run uvicorn realtime_server:app --port 3005
     uv run pytest tests/
     ```
   
   - **Option 2: Manual activation** - Traditional approach
     ```bash
     source .venv/bin/activate  # On Linux/macOS
     .venv\Scripts\activate     # On Windows
     # Now run commands normally:
     python script.py
     uvicorn realtime_server:app --port 3005
     deactivate  # When finished
     ```

3. **Configure Environment Variables**

   Brainwave requires the OpenAI API key to function. Set the `OPENAI_API_KEY` environment variable:

   - **On macOS/Linux:**

     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```

   - **On Windows (Command Prompt):**

     ```cmd
     set OPENAI_API_KEY=your-openai-api-key
     ```

   - **On Windows (PowerShell):**

     ```powershell
     $env:OPENAI_API_KEY="your-openai-api-key"
     ```

   Optionally, you can configure where Brainwave saves audio recordings and transcripts by setting the `BRAINWAVE_RECORDINGS_DIR` environment variable:

   - **On macOS/Linux:**
     ```bash
     export BRAINWAVE_RECORDINGS_DIR='/path/to/your/recordings'
     ```

   - **On Windows (Command Prompt):**
     ```cmd
     set BRAINWAVE_RECORDINGS_DIR=C:\Users\YourUsername\Dropbox\Stable\AudioWrite
     ```

   - **On Windows (PowerShell):**
     ```powershell
     $env:BRAINWAVE_RECORDINGS_DIR="C:\Users\YourUsername\Dropbox\Stable\AudioWrite"
     ```

   If not set, recordings will be saved to a `recordings` directory in the project folder.

   Brainwave now writes a timestamped fail-safe WAV for every session, even if
   the realtime OpenAI connection fails or drops mid-stream. These safeguard files
   share the same directory (or your custom `BRAINWAVE_RECORDINGS_DIR`) and are
   named with the session's time tag (for example, `20240805_132233.wav`). Once
   the descriptive WAV _and_ matching transcript are written successfully, the
   temporary timestamp file is cleaned up automatically; otherwise it stays put
   as your fail-safe copy.

4. **Launch the Server**

   Start the FastAPI server using uv:

   ```bash
   uv run uvicorn realtime_server:app --host 0.0.0.0 --port 3005
   ```

   **Alternative hosting options:**
   ```bash
   # For local development only (localhost access)
   uv run uvicorn realtime_server:app --port 3005

   # For production with custom host and port
   uv run uvicorn realtime_server:app --host 0.0.0.0 --port 8000

   # With auto-reload for development
   uv run uvicorn realtime_server:app --host 0.0.0.0 --port 3005 --reload
   ```

   The server will be accessible at:
   - **Local access:** `http://localhost:3005`
   - **Network access:** `http://[your-machine-ip]:3005` (replace with your actual IP address for access from other devices)

   > **Note:** `uv run` automatically activates the virtual environment and runs the command with all dependencies available. No manual activation required!

5. **Access the Application**

   Open your web browser and navigate to `http://localhost:3005` to interact with Brainwave's speech recognition interface.

### Hosting Troubleshooting

If you encounter issues with hosting:

- **Port already in use:** Change the port number (e.g., `--port 3006`)
- **Permission denied on port 80/443:** Use a port number above 1024
- **Can't access from other devices:** Ensure you're using `--host 0.0.0.0` and check your firewall settings
- **Environment variables not found:** Make sure `OPENAI_API_KEY` is set in your shell session

### Browser Troubleshooting

**Microsoft Edge Installed App Issues:**

If Brainwave is installed as an app in Microsoft Edge, it may occasionally stop recording audio. When this happens:

1. **Verify the fix will work:** Open Brainwave in an Edge private/incognito window. If it works there, this confirms the issue is with the installed app.
2. **Solution:** Uninstall the installed app and reinstall it. This will restore recording functionality.

**Sound Wave Indicator Issues:**

The sound wave indicator may not work properly in Firefox. If the indicator doesn't animate during recording (but recording itself works), switch to Chrome or Microsoft Edge where the indicator has been tested and works reliably.

### Additional uv Commands

Here are useful uv commands for managing your development environment:

**Environment Management:**
```bash
# Check uv version
uv --version

# Show project information
uv tree  # Show dependency tree
uv pip list  # List installed packages

# Clean and refresh environment
uv sync --reinstall  # Reinstall all packages
uv cache clean       # Clear uv cache
```

**Dependency Management:**
```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --group dev package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv lock --upgrade      # Update lock file with latest versions
uv sync --upgrade      # Sync with updated lock file
```

**Running Commands:**
```bash
# Run Python commands (recommended approach)
uv run python script.py
uv run python -m module_name
uv run pytest tests/
uv run uvicorn realtime_server:app --port 3005

# Alternative: Manual activation (traditional approach)
source .venv/bin/activate  # On Linux/macOS
.venv\Scripts\activate     # On Windows
# Then run commands normally without 'uv run'
```

**Project Setup Commands:**
```bash
# Initialize a new uv project
uv init

# Install dependencies from existing pyproject.toml
uv sync

# Install specific Python version for project
uv python install 3.12
```

---

## Code Structure & Architecture

Understanding the architecture of **Brainwave** provides insights into its real-time processing capabilities and multilingual support. The project is organized into several key components, each responsible for distinct functionalities.

### 1. **Backend**

#### a. `realtime_server.py`

- **Framework:** Utilizes **FastAPI** to handle HTTP and WebSocket connections, offering high performance and scalability.
- **WebSocket Endpoint:** Establishes a `/ws` endpoint for real-time audio streaming between the client and server.
- **Audio Processing:**
  - **`AudioProcessor` Class:** Resamples incoming audio data from 48kHz to 24kHz to match OpenAI's requirements.
  - **Buffer Management:** Accumulates audio chunks for efficient processing and transmission.
- **Concurrency:** Employs `asyncio` to manage asynchronous tasks for receiving and sending audio data, ensuring non-blocking operations.
- **Logging:** Implements comprehensive logging to monitor connections, data flow, and potential errors.

#### b. `openai_realtime_client.py`

- **WebSocket Client:** Manages the connection to OpenAI's real-time API, facilitating the transmission of audio data and reception of transcriptions.
- **Session Management:** Handles session creation, updates, and closure, ensuring a stable and persistent connection.
- **Event Handlers:** Registers and manages handlers for various message types from OpenAI, allowing for customizable responses and actions based on incoming data.
- **Error Handling:** Incorporates robust mechanisms to handle and log connection issues or unexpected messages.

#### c. `prompts.py`

- **Prompt Definitions:** Contains a dictionary of prompts in both Chinese and English, tailored for tasks such as paraphrasing, readability enhancement, and generating insightful summaries.
- **Customization:** Allows for easy modification and extension of prompts to cater to different processing requirements or languages.

### 2. **Frontend**

#### a. `static/realtime.html`

- **User Interface:** Provides a clean and responsive UI for users to interact with Brainwave, featuring:
  - **Recording Controls:** A toggle button to start and stop audio recording.
  - **Transcript Display:** A section to display the transcribed and summarized text in real-time.
  - **Copy Functionality:** Enables users to easily copy the summarized text.
  - **Timer:** Visual feedback to indicate recording duration.

- **Styling:** Utilizes CSS to ensure a modern and user-friendly appearance, optimized for both desktop and mobile devices.

- **Audio Handling:**
  - **Web Audio API:** Captures audio streams from the user's microphone, processes them into the required format, and handles chunking for transmission.
  - **WebSocket Integration:** Establishes and manages the WebSocket connection to the backend server, ensuring seamless data flow.

### 3. **Configuration**

#### a. `pyproject.toml`

Defines the project configuration, dependencies, and metadata following modern Python packaging standards:
- **Project metadata**: Name, version, description, and Python version requirements
- **Dependencies**: Runtime dependencies needed for the application
- **Development dependencies**: Testing and development tools organized in dependency groups
- **Build configuration**: Settings for package building using hatchling

### 4. **Prompts & Text Processing**

Brainwave leverages a suite of predefined prompts to enhance text processing capabilities:

- **Paraphrasing:** Corrects speech-to-text errors and improves punctuation without altering the original meaning.
- **Readability Enhancement:** Improves the readability of transcribed text by adding appropriate punctuation and formatting.
- **Summary Generation:** Creates concise and logical summaries from the user's spoken input, making ideas easier to review and manage.

These prompts are meticulously crafted to ensure that the transcribed text is not only accurate but also contextually rich and user-friendly.

### 5. **Logging & Monitoring**

Comprehensive logging is integrated throughout the backend components to monitor:

- **Connection Status:** Tracks WebSocket connections and disconnections.
- **Data Transmission:** Logs the size and status of audio chunks being processed and sent.
- **Error Reporting:** Captures and logs any errors or exceptions, facilitating easier debugging and maintenance.

---

## Testing

Brainwave includes a comprehensive test suite to ensure reliability and maintainability. The tests cover various components:

- **Audio Processing Tests:** Verify the correct handling of audio data, including resampling and buffer management.
- **LLM Integration Tests:** Test the integration with language models (GPT and Gemini) for text processing.
- **API Endpoint Tests:** Ensure the FastAPI endpoints work correctly, including streaming responses.
- **WebSocket Tests:** Verify real-time communication for audio streaming.

To run the tests:

1. **Install Test Dependencies**

   Test dependencies are automatically installed when you run `uv sync` (they're included in the `dev` dependency group).

2. **Run Tests**

   ```bash
   # Run all tests
   uv run pytest tests/

   # Run tests with verbose output
   uv run pytest -v tests/

   # Run tests for a specific component
   uv run pytest tests/test_audio_processor.py
   ```

3. **Test Environment**

   Tests use mocked API clients to avoid actual API calls. Set up the test environment variables:
   ```bash
   export OPENAI_API_KEY='test_key'  # For local testing
   export GOOGLE_API_KEY='test_key'  # For local testing
   ```

The test suite is designed to run without making actual API calls, making it suitable for CI/CD pipelines.

---

## Conclusion

**Brainwave** revolutionizes the way users capture and organize their ideas by providing a seamless speech recognition and summarization tool. Its real-time processing capabilities, combined with multilingual support and sophisticated text enhancement, make it an invaluable asset for anyone looking to efficiently manage their thoughts and ideas. Whether you're brainstorming, taking notes, or organizing project ideas, Brainwave ensures that your spoken words are transformed into clear, organized, and actionable summaries.

For any questions, contributions, or feedback, feel free to [open an issue](https://github.com/grapeot/brainwave/issues) or submit a pull request on the repository.

---

*Empower Your Ideas with Brainwave!*
