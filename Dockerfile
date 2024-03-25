FROM node:14

WORKDIR /Blueprint
COPY package.json .
RUN npm install
COPY . .
CMD npm start