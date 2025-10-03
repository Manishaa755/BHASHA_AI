// bhasha.js — simple client SDK
export class BhashaClient {
  constructor(apiKey, baseUrl = "http://127.0.0.1:8000") {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async _post(path, body, isJson = true) {
    const headers = { "X-API-Key": this.apiKey };
    let opts;
    if (isJson) {
      headers["Content-Type"] = "application/json";
      opts = { method: "POST", headers, body: JSON.stringify(body) };
    } else {
      opts = { method: "POST", headers, body };
    }
    const r = await fetch(this.baseUrl + path, opts);
    if (r.status === 429) throw new Error("Rate limit exceeded");
    if (!r.ok) {
      const msg = await r.text();
      throw new Error(`API error ${r.status}: ${msg}`);
    }
    return r;
  }

  async stt(file) {
    const fd = new FormData();
    fd.append("file", file);
    const r = await this._post("/tool/stt", fd, false);
    return r.json();
  }

  async tts(text, { voice = "alloy", format = "mp3", model = "tts-1" } = {}) {
    const r = await this._post("/tool/tts", { text, voice, format, tts_model: model });
    const buf = await r.arrayBuffer();
    return new Blob([buf], { type: r.headers.get("content-type") });
  }

  async s2s(file, { target_lang = "auto", voice = "alloy", format = "mp3" } = {}) {
    const fd = new FormData();
    fd.append("file", file);
    const url = `/tool/s2s?target_lang=${encodeURIComponent(target_lang)}&voice=${voice}&format=${format}`;
    const r = await this._post(url, fd, false);
    const buf = await r.arrayBuffer();
    return new Blob([buf], { type: r.headers.get("content-type") });
  }

  async reasonDoc({ task = "auto", question, target_lang = "auto", text, file } = {}) {
    const fd = new FormData();
    fd.append("task", task);
    if (question) fd.append("question", question);
    fd.append("target_lang", target_lang);
    if (text) fd.append("text", text);
    if (file) fd.append("file", file);
    const r = await this._post("/tool/reason-doc", fd, false);
    return r.json();
  }

  async langDetect({ text, file } = {}) {
    const fd = new FormData();
    if (text) fd.append("text", text);
    if (file) fd.append("file", file);
    const r = await this._post("/tool/lang-detect", fd, false);
    return r.json();
  }
}
