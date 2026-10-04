const start=document.querySelector('#start'),stop=document.querySelector('#stop'),status=document.querySelector('#status'),transcript=document.querySelector('#transcript');
let socket,context,microphone,node,controls={},outstanding=0;
async function cleanup(message='Stopped'){
  microphone?.getTracks().forEach(t=>t.stop());microphone=null;
  node?.disconnect();node=null;
  const oldContext=context;context=null;
  if(socket){socket.onclose=null;socket.onerror=null;socket.close();socket=null;}
  if(oldContext && oldContext.state!=='closed'){await oldContext.close();}
  outstanding=0;start.disabled=false;stop.disabled=true;status.textContent=message;
}
start.onclick=async()=>{
  start.disabled=true;status.textContent='Starting…';transcript.textContent='';
  try{
    if(!navigator.mediaDevices?.getUserMedia){throw new Error('Microphone access needs localhost or HTTPS. For a remote GPU, use SSH port forwarding and open http://localhost:8000.');}
    microphone=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:false}});
    context=new AudioContext();await context.resume();
    await context.audioWorklet.addModule('/static/audio.js');
    node=new AudioWorkletNode(context,'duplex-audio');
    socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/stream`);socket.binaryType='arraybuffer';
    socket.onmessage=({data})=>{
      if(data instanceof ArrayBuffer){node?.port.postMessage({pcm:data},[data]);return;}
      const m=JSON.parse(data);
      if(m.type==='ready'){
        controls=m.control_tokens;
        context.createMediaStreamSource(microphone).connect(node);node.connect(context.destination);
        status.textContent=`Listening · up to ${Math.floor(m.max_conversation_seconds)} seconds per conversation`;stop.disabled=false;
      }else if(m.type==='ack'){outstanding-=m.samples;}
      else if(m.type==='error'){cleanup(m.message);}
      else if(m.type==='event'){
        if(m.interrupt || m.token_id===controls.interrupt){node?.port.postMessage({clear:true});}
        if(m.response || m.token_id===controls.response){transcript.textContent+='\n';}
        if(m.text_delta){transcript.textContent+=m.text_delta;}
      }
    };
    socket.onclose=()=>cleanup('Conversation ended');
    socket.onerror=()=>cleanup('Could not connect to the model');
    node.port.onmessage=({data})=>{
      if(socket?.readyState!==WebSocket.OPEN)return;
      outstanding+=data.byteLength/2;
      if(outstanding>32000){cleanup('The model is falling behind. Start a new conversation.');return;}
      socket.send(data);
    };
  }catch(error){await cleanup(error.message);}
};
stop.onclick=()=>cleanup();
