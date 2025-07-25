FROM mcr.microsoft.com/playwright/python:v1.53.0-noble

# Set a working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app
RUN pip install -r requirements.txt

ENTRYPOINT []


# Ensure Node.js, npm (and npx) are set up
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
RUN apt-get install -y nodejs

ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
RUN npm install playwright@1.53.0 -g
RUN npx playwright@1.53.0 install

CMD ["npx", "-y", "playwright@1.53.0", "run-server", "--port", "3000", "--host", "0.0.0.0"]
# CMD ["xvfb-run", "--auto-servernum", "--server-args=-screen 0 1280x1024x24", "npx", "-y", "playwright@1.53.0", "run-server", "--port", "3000", "--host", "0.0.0.0"]
