const wrapper = document.querySelector(".wrapper");
const loginLink = document.querySelector(".login-link");
const registerLink = document.querySelector(".register-link");
const btnPopup = document.querySelector(".btnLogin-popup");
const iconClose = document.querySelector(".icon-close");
const btnClose = document.querySelector(".btn-close");
btnPopup.addEventListener("click", () => {
  wrapper.classList.remove("active-popup");
});
registerLink.addEventListener("click", () => {
  wrapper.classList.add("active");
});
loginLink.addEventListener("click", () => {
  wrapper.classList.remove("active");
});
iconClose.addEventListener("click", () => {
  wrapper.classList.remove("active-popup");
});
btnPopup.addEventListener("click", () => {
  wrapper.classList.add("active-popup");
});
document.getElementById('loginForm').addEventListener('submit', function (e) {
  e.preventDefault(); 
document.getElementById('loginForm').addEventListener('submit', function (e) {
  e.preventDefault(); 
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;
  if (email === "user@example.com" && password === "password123") {
      alert("Login successful!");
      window.location.href = "website.html";
  } else {
      alert("Invalid email or password. Please try again.");
  }
})
});

