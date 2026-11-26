// 웹 버전(index.html)의 스크립트를 Node.js에서 실행해 질문마다 계획과 답(HTML)을 돌려준다.
//   node tests/js_harness.js index.html < 질문목록.json
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.match(/<script>([\s\S]*)<\/script>/)[1];
const el = () => ({className: "", innerHTML: "", textContent: "", type: "", scrollTop: 0, scrollHeight: 0,
                   appendChild() {}, addEventListener() {}, focus() {}});
globalThis.document = {getElementById: el, createElement: el};
const questions = JSON.parse(fs.readFileSync(0, "utf8"));
// 화면을 처음 열 때의 예시 질문이 바꾼 '직전 데이터'를 파이썬 시작 상태(첫 데이터)로 되돌린 뒤 묻는다
const run = new Function("questions", src + `
  last = DATA.length ? DATA[0].spec.title : null; prevPlan = null;
  return questions.map(x => { const q = typeof x === "string" ? x : x.q;
    if (x.fresh) prevPlan = null;
    lastPlan = null; const html = respond(q); return {plan: lastPlan, html}; });`);
process.stdout.write(JSON.stringify(run(questions)));
