require('dotenv').config();
const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const mediasoup = require('mediasoup');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

let worker; let router;
(async () => {
  worker = await mediasoup.createWorker({ rtcMinPort: 20000, rtcMaxPort: 20200 });
  router = await worker.createRouter({ mediaCodecs: [{ kind: 'audio', mimeType: 'audio/opus', clockRate: 48000, channels: 2 }] });

  io.on('connection', socket => {
    socket.on('createTransport', async (_, cb) => {
      const transport = await router.createWebRtcTransport({
        listenIps: [{ ip: '0.0.0.0', announcedIp: process.env.ANNOUNCED_IP || null }],
        enableTcp: true, enableUdp: true, preferUdp: true,
      });
      cb({ id: transport.id, iceParameters: transport.iceParameters, iceCandidates: transport.iceCandidates, dtlsParameters: transport.dtlsParameters });
      socket.on('connectTransport', async ({ dtlsParameters }) => { await transport.connect({ dtlsParameters }); });
      socket.on('produce', async ({ kind, rtpParameters }, cb2) => { const producer = await transport.produce({ kind, rtpParameters }); cb2({ id: producer.id }); });
    });
  });

  server.listen(process.env.PORT || 3001, ()=> console.log('mediasoup running'));
})();
