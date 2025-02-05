FROM node:20-slim

WORKDIR /app

# Install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy application code
COPY frontend/ .

# Build the application
RUN npm run build

# Install serve to run the built application
RUN npm install -g serve

# Expose the port
EXPOSE 3000

# Start the application in development mode
CMD ["npm", "run", "dev"]
