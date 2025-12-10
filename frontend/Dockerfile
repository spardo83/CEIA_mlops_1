FROM node:22-alpine
WORKDIR /app

COPY package*.json ./
RUN npm install
RUN rm -f package-lock.json

COPY . .

EXPOSE 5173

# Vite requires --host to bind to 0.0.0.0
CMD ["npm", "run", "dev", "--", "--host"]
