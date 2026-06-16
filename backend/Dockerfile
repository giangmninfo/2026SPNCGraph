FROM python:3.11-slim

# 1. Set the main working directory
WORKDIR /app

# 2. Copy requirements into a 'backend' folder
# We do this to match the structure your code expects
COPY Requirements-Full.txt /app/backend/Requirements-Full.txt

# 3. Install dependencies
RUN pip install --no-cache-dir -r /app/backend/Requirements-Full.txt

# 4. Copy ALL your code into the 'backend' folder
COPY . /app/backend

# 5. CRITICAL: Tell Python that '/app' is the root
ENV PYTHONPATH=/app

# 6. Expose the port
EXPOSE 8000

# 7. Run the app pointing to the correct location
CMD ["python", "backend/app.py"]