/* Vue 3 progressive enhancement for same-origin website previews. */
(function(){
 'use strict';
 const menu=document.getElementById('menu-toggle'), nav=document.getElementById('nav-links');
 if(menu&&nav){const close=()=>{nav.classList.remove('open');menu.setAttribute('aria-expanded','false');menu.textContent='☰';};menu.addEventListener('click',()=>{const open=nav.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));menu.textContent=open?'×':'☰';});nav.querySelectorAll('a').forEach(a=>a.addEventListener('click',close));document.addEventListener('keydown',e=>{if(e.key==='Escape')close();});}
 document.querySelectorAll('.copy-button').forEach(b=>b.addEventListener('click',async()=>{const code=b.parentElement.querySelector('code');if(!code)return;try{await navigator.clipboard.writeText(code.textContent);b.textContent='Copied ✓';}catch(e){const sel=window.getSelection(),r=document.createRange();r.selectNodeContents(code);sel.removeAllRanges();sel.addRange(r);b.textContent='Select & copy';}}));
 const options={portfolio:{title:'AI & Generative AI Portfolio',url:'https://su394611-del.github.io/'},mixtral:{title:'Mixtral Healthcare LLM Project',url:'https://su394611-del.github.io/Mixtral-Healthcare-LLM-From-Scratch/'},rag:{title:'RAG Retrieval Service Project',url:'https://su394611-del.github.io/RAG-service/'}};
 const root=document.getElementById('explorer-app'); if(!root)return;
 function fallback(){ const cards=root.querySelectorAll('[data-preview]'),viewer=root.querySelector('#preview-frame'),empty=root.querySelector('#preview-placeholder'),label=root.querySelector('#preview-label'),link=root.querySelector('#preview-open-link'),close=root.querySelector('#preview-close');let active='';
 function select(k){const opt=options[k];if(!opt)return;active=k; viewer.hidden=false;viewer.parentElement.style.display='block';viewer.src=opt.url;viewer.title='Preview of '+opt.title;empty.hidden=true;label.textContent=opt.title;link.href=opt.url;link.hidden=false;link.style.display='inline-flex';close.hidden=false;close.style.display='inline-flex';cards.forEach(c=>c.classList.toggle('active',c.dataset.preview===k));}
 cards.forEach(b=>b.addEventListener('click',()=>select(b.dataset.preview)));
 close?.addEventListener('click',()=>{viewer.removeAttribute('src');viewer.hidden=true;viewer.parentElement.style.display='none';empty.hidden=false;close.hidden=true;close.style.display='none';link.hidden=true;link.style.display='none';label.textContent='Choose a project preview';cards.forEach(c=>c.classList.remove('active'));});
 }
 if(window.Vue&&typeof window.Vue.createApp==='function'){
  const {createApp}=window.Vue;
  try{createApp({data(){return{active:'',items:options};},computed:{selected(){return this.items[this.active]||null;}},methods:{open(k){this.active=k;},close(){this.active='';}}}).mount(root);return;}catch(e){console.warn('Vue preview unavailable, using native controls.',e);}
 }
 root.removeAttribute('v-cloak');fallback();
})();
