const socket = io();

let messageContainer = document.querySelector(".messages");


const messageInput = document.getElementById("messageInput")
const qqq = document.getElementById("qqq")
const config = document.getElementById("config")
const init2= document.getElementById("init2")
const scan=document.getElementById("scan")
const initialise=document.getElementById("initialise")
const calc_vol=document.getElementById("calc_vol")
config.addEventListener("click", () => {
    socket.emit("config")
})
init2.addEventListener("click",()=>{
    socket.emit("init2")
    qqq.innerHTML="420"
})
calc_vol.addEventListener("click",()=>{
    socket.emit("calc_vol")
})
initialise.addEventListener("click", () => {
    socket.emit("initialise")
    qqq.innerHTML=""
})
socket.on("initialise", (c) => {
    qqq.innerHTML = c;

})
socket.on("calc_vol", (c)=>{
    let messageElement = document.createElement("p")
    messageElement.innerText = c
    messageContainer.appendChild(messageElement)
})
socket.on("qwer", (c)=>{
    qqq.innerHTML=c;
})