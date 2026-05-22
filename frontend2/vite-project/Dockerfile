# Stage 1 — Build the React app
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code and build
COPY . .
RUN npm run build

# Stage 2 — Serve with Nginx
FROM nginx:alpine

# Copy the built React files into Nginx's serve folder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx config (handles API proxy + React routing)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]