FROM python:3.7-slim
WORKDIR /app
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY start.sh /usr/bin/start.sh
RUN chmod +x /usr/bin/start.sh
CMD ["/usr/bin/start.sh"]