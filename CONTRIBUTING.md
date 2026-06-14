# Contributing to this project

Hi! Thanks for wanting to help out. This project is about making it easy to fine-tune AI models on local devices without sharing private data. I want to keep it simple and accessible to everyone.

Here is a simple guide on how you can contribute.

## How you can help

You can help in many ways:
- Fix bugs or errors in the code.
- Make the Web UI look better or add new interactive features.
- Write better documentation or add tutorials.
- Test the platform on different hardware (like Mac, Windows, Linux) and report how it runs.

## Local setup

To work on the code locally:

1. Fork this repository on GitHub and clone it to your computer.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete project locally to make sure everything works:
   ```bash
   python run_platform.py --mode all
   ```

## Making changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b my-new-feature
   ```
2. Make your changes in the code. Keep your code clean and use simple logic.
3. Test your changes locally. Make sure the server and the client run without throwing any errors.
4. Commit your changes with a simple, clear message:
   ```bash
   git commit -m "Fix import path error in client connection"
   ```
5. Push to your branch and open a Pull Request (PR) on GitHub.

## Code Style

- Write clean, simple Python code.
- Avoid using complex libraries if simple standard Python libraries can do the job.
- Comment your code if you are doing something that is not obvious.
- For the Web UI, use standard HTML, CSS, and Javascript. Try to keep the design responsive so it works on both mobile and desktop screens.

If you have any questions or need help, just open an issue on GitHub and we will try to reply as soon as possible.
