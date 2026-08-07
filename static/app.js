/* Everline voice call client.
 *
 * Captures the microphone with an AudioWorklet, downsamples to 16 kHz PCM16,
 * and streams it to the server over a WebSocket. Agent replies arrive as MP3
 * binary frames and are played back. The link is half-duplex: while the agent
 * is speaking, mic frames are dropped so the agent never hears itself.
 */

const orb = document.getElementById("orb");
const statusLabel = document.getElementById("status-label");
const callBtn = document.getElementById("call-btn");
const callBtnLabel = document.getElementById("call-btn-label");
const transcript = document.getElementById("transcript");

let ws = null;
let audioCtx = null;
let mediaStream = null;
let workletNode = null;
let onCall = false;
let agentSpeaking = false;
let currentAudio = null;

const WORKLET_CODE = `
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0;
    this.buf = [];
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // Simple decimation with linear interpolation to 16 kHz.
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        this.buf.push(ch[i]);
      }
    }
    if (this.buf.length >= 640) { // 40 ms @ 16 kHz
      const out = new Int16Array(this.buf.length);
      for (let i = 0; i < this.buf.length; i++) {
        const s = Math.max(-1, Math.min(1, this.buf[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      this.port.postMessage(out.buffer, [out.buffer]);
      this.buf = [];
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCapture);
`;

function setState(state, label) {
  orb.className = "orb " + state;
  statusLabel.textContent = label;
}

function addBubble(role, text) {
  const hint = transcript.querySelector(".hint");
  if (hint) hint.remove();
  const div = document.createElement("div");
  div.className = "bubble " + role;
  div.innerHTML = `<span class="who">${role === "agent" ? "Riley · Dispatcher" : "You"}</span>`;
  div.appendChild(document.createTextNode(text));
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
}

async function startCall() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    setState("idle", "Microphone access denied");
    return;
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/call`);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    audioCtx = new AudioContext();
    const blob = new Blob([WORKLET_CODE], { type: "application/javascript" });
    await audioCtx.audioWorklet.addModule(URL.createObjectURL(blob));
    const source = audioCtx.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-capture");
    workletNode.port.onmessage = (e) => {
      if (ws && ws.readyState === WebSocket.OPEN && !agentSpeaking) {
        ws.send(e.data);
      }
    };
    source.connect(workletNode);

    onCall = true;
    callBtn.classList.add("on-call");
    callBtnLabel.textContent = "End Call";
    setState("listening", "Connecting you to Riley...");
  };

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "transcript") {
        addBubble(msg.role, msg.text);
      } else if (msg.type === "state") {
        if (msg.value === "listening") setState("listening", "Listening — go ahead");
        else if (msg.value === "thinking") setState("thinking", "Riley is thinking...");
        else if (msg.value === "speaking") setState("speaking", "Riley is speaking");
      }
    } else {
      playAgentAudio(event.data);
    }
  };

  ws.onclose = () => endCall(false);
  ws.onerror = () => endCall(false);
}

function playAgentAudio(arrayBuffer) {
  const blob = new Blob([arrayBuffer], { type: "audio/mpeg" });
  const url = URL.createObjectURL(blob);
  agentSpeaking = true;
  setState("speaking", "Riley is speaking");
  currentAudio = new Audio(url);
  currentAudio.onended = () => {
    agentSpeaking = false;
    URL.revokeObjectURL(url);
    if (onCall) setState("listening", "Listening — go ahead");
  };
  currentAudio.onerror = () => {
    agentSpeaking = false;
    if (onCall) setState("listening", "Listening — go ahead");
  };
  currentAudio.play();
}

function endCall(sendClose = true) {
  if (!onCall && !ws) return;
  onCall = false;
  agentSpeaking = false;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if (ws) {
    if (sendClose && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "end_call" }));
    }
    ws.close();
    ws = null;
  }
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  callBtn.classList.remove("on-call");
  callBtnLabel.textContent = "Start Call";
  setState("idle", "Call ended — press Start Call to ring again");
}

callBtn.addEventListener("click", () => {
  if (onCall) endCall();
  else startCall();
});
