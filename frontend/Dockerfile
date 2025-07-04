# 构建阶段
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
# 指定华为源
RUN npm config set registry https://mirrors.huaweicloud.com/repository/npm/
RUN npm install
COPY . .
RUN npm run build -- --mode production

# 生产阶段
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80