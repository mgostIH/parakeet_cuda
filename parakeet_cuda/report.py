# SPDX-License-Identifier: Apache-2.0
"""Write a listening report; embed small audio and link large recordings."""
import base64
import html
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import quote


def write_report(data,audio,output):
    audio,output=Path(audio).resolve(),Path(output)
    if audio.stat().st_size<=10*1024**2:
        mime=mimetypes.guess_type(audio.name)[0] or 'application/octet-stream'
        encoded=base64.b64encode(audio.read_bytes()).decode('ascii')
        source=f'data:{mime};base64,{encoded}'
        note='Audio is embedded in this report.'
    else:
        source=quote(os.path.relpath(audio,output.resolve().parent),safe='/')
        note='Audio is linked to the original file; keep it available at its current location.'
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    document='''<!doctype html><meta charset="utf-8"><title>Speaker transcript</title>
<style>
body{font:17px/1.6 system-ui;max-width:960px;margin:40px auto;padding:0 24px;color:#20252b;background:#fafafa}
h1{font-size:26px}.controls{position:sticky;top:0;background:#fafafa;padding:16px 0;z-index:2}audio{width:100%}
.turn{display:block;width:100%;text-align:left;border:1px solid #ddd;border-left:5px solid var(--color);background:white;padding:12px 16px;margin:12px 0;border-radius:8px;cursor:pointer;font:inherit}
.turn:hover,.turn.playing{background:#eef4fc}.meta{font-size:14px;color:#586270}canvas{width:100%;height:75px;background:white;border-radius:8px}.note{font-size:14px;color:#586270}
</style><h1>Speaker transcript</h1>
<p class="note">Click a turn to listen from its start. Speaker labels describe voices in this recording. Text is the original Parakeet transcription.</p>
<p class="note">__AUDIO_NOTE__</p>
<div class="controls"><audio id="player" controls src="__AUDIO_SOURCE__"></audio><canvas id="timeline" height="180" width="1000"></canvas></div>
<div id="turns"></div><script>
const data=__DATA__,player=document.getElementById('player'),turns=document.getElementById('turns');
const colors={speaker_1:'#2878b8',speaker_2:'#d76d25',speaker_3:'#7c4ca7',speaker_4:'#548a45',speaker_5:'#be4980',speaker_6:'#918028',speaker_7:'#24918c',speaker_8:'#675bc6',unknown:'#8b929a'};
const duration=data.pcm.duration,canvas=document.getElementById('timeline'),ctx=canvas.getContext('2d');
const format=t=>Math.floor(t/60)+':'+(t%60).toFixed(2).padStart(5,'0');
for(const turn of data.speaker_turns){
 const button=document.createElement('button');button.className='turn';button.style.setProperty('--color',colors[turn.speaker]||'#51856c');
 const meta=document.createElement('div');meta.className='meta';meta.textContent=format(turn.start)+' – '+format(turn.end)+' · '+turn.speaker.replace('_',' ');
 const text=document.createElement('div');text.textContent=turn.text;button.append(meta,text);turns.append(button);
 button.onclick=()=>{player.currentTime=Math.max(0,turn.start-.15);player.play()};turn.button=button;
}
function draw(){ctx.clearRect(0,0,1000,180);for(const segment of data.diarization.segments){
 const row=Number(segment.speaker.split('_')[1])-1;ctx.fillStyle=colors[segment.speaker]||'#51856c';
 ctx.fillRect(segment.start/duration*1000,row*22+4,(segment.end-segment.start)/duration*1000,16);}
 ctx.fillStyle='#20252b';ctx.fillRect(player.currentTime/duration*1000,0,2,180);}
player.ontimeupdate=()=>{draw();for(const turn of data.speaker_turns)turn.button.classList.toggle('playing',player.currentTime>=turn.start&&player.currentTime<turn.end)};
canvas.onclick=event=>{player.currentTime=(event.clientX-canvas.getBoundingClientRect().left)/canvas.clientWidth*duration};draw();
</script>'''.replace('__AUDIO_NOTE__',note).replace('__AUDIO_SOURCE__',html.escape(source,quote=True)).replace('__DATA__',payload)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(document)
    return output
