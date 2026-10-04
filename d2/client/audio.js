class DuplexAudio extends AudioWorkletProcessor {
  constructor() {
    super(); this.pending=[]; this.offset=0; this.phase=0; this.previous=0; this.input=[];
    this.port.onmessage=({data})=>{
      if(data.clear){this.pending=[];this.offset=0;this.outputPhase=0;this.lastOutput=0;this.nextOutput=0;}
      else if(data.pcm){this.pending.push(new Int16Array(data.pcm));}
    };
    this.outputPhase=0;this.lastOutput=0;this.nextOutput=0;
  }
  nextSample(){
    while(this.pending.length && this.offset>=this.pending[0].length){this.pending.shift();this.offset=0;}
    return this.pending.length ? this.pending[0][this.offset++]/32768 : 0;
  }
  process(inputs,outputs){
    const mic=inputs[0]?.[0];
    if(mic) for(const current of mic){
      while(this.phase<1){
        const v=this.previous+(current-this.previous)*this.phase;
        this.input.push(Math.max(-32768,Math.min(32767,Math.round(v*32768))));
        this.phase+=sampleRate/16000;
      }
      this.phase-=1;this.previous=current;
      if(this.input.length>=640){
        const pcm=new Int16Array(this.input.splice(0,640));
        this.port.postMessage(pcm.buffer,[pcm.buffer]);
      }
    }
    const output=outputs[0][0];
    for(let i=0;i<output.length;i++){
      output[i]=this.lastOutput+(this.nextOutput-this.lastOutput)*this.outputPhase;
      this.outputPhase+=24000/sampleRate;
      while(this.outputPhase>=1){this.outputPhase-=1;this.lastOutput=this.nextOutput;this.nextOutput=this.nextSample();}
    }
    return true;
  }
}
registerProcessor('duplex-audio',DuplexAudio);
