(function(global){
  function _get(url){ return fetch(url).then(r=>r.json()); }
  function createFloatingButton(){
    const btn = document.createElement('button');
    btn.innerText = 'Voice AI';
    btn.style.position='fixed'; btn.style.right='24px'; btn.style.bottom='24px'; btn.style.zIndex=99999;
    document.body.appendChild(btn);
    return btn;
  }
  async function init(opts){
    const btn = createFloatingButton();
    const panel = document.createElement('div');
    panel.style.position='fixed'; panel.style.right='24px'; panel.style.bottom='80px';
    panel.style.width='360px'; panel.style.height='420px'; panel.style.zIndex=99999;
    panel.style.background='white'; panel.style.border='1px solid #ddd';
    panel.style.padding='12px'; panel.style.overflow='auto'; panel.style.display='none';
    document.body.appendChild(panel);

    btn.onclick = ()=> { panel.style.display = panel.style.display==='none' ? 'block': 'none'; }

    const status = document.createElement('div'); panel.appendChild(status);
    const transcriptBox = document.createElement('pre'); transcriptBox.style.height='250px'; transcriptBox.style.overflow='auto'; panel.appendChild(transcriptBox);

    const startBtn = document.createElement('button'); startBtn.innerText='Start Recording'; panel.appendChild(startBtn);
    const stopBtn = document.createElement('button'); stopBtn.innerText='Stop'; stopBtn.disabled=true; panel.appendChild(stopBtn);

    let ws; let mediaRecorder;
    startBtn.onclick = async ()=>{
      status.innerText = 'Requesting token...';
      const tokenResp = await _get(opts.tokenEndpoint + '?user_id=' + encodeURIComponent(opts.userId || 'anon'));
      const token = tokenResp.token;
      status.innerText = 'Connecting...';
      ws = new WebSocket((opts.wsUrl||'ws://localhost:8000') + '/ws/stream');
      ws.binaryType = 'arraybuffer';
      ws.onopen = async ()=>{
        ws.send(token);
        status.innerText = 'Connected, requesting mic...';
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e)=>{ if (e.data.size>0){ e.data.arrayBuffer().then(buf=> ws.send(buf)); } };
        mediaRecorder.start(250);
        startBtn.disabled=true; stopBtn.disabled=false; status.innerText='Recording...';
      };
      ws.onmessage = (ev)=>{
        try{
          const data = JSON.parse(ev.data);
          if (data.type === 'transcript.partial'){ transcriptBox.innerText = data.text + '\\n' + transcriptBox.innerText; }
          if (data.type === 'ai.response'){ transcriptBox.innerText = 'AI: ' + data.text + '\\n' + transcriptBox.innerText; if (data.tts_b64){ new Audio('data:audio/mpeg;base64,'+data.tts_b64).play(); } }
        }catch(e){ console.log('msg', ev.data); }
      };
    };
    stopBtn.onclick = ()=>{ if (mediaRecorder){ mediaRecorder.stop(); } if (ws) ws.send('__end_audio__'); stopBtn.disabled=true; startBtn.disabled=false; status.innerText='Stopped'; };
  }
  global.MultilingualWidget = { init };
})(window);
