const ADMIN_PASSWORD = "admin123";

function promptAdminPassword() {
    const password = prompt("Enter Admin Password:");
    if (password !== ADMIN_PASSWORD) {
        alert("Incorrect password!");
        return false;
    }
    return true;
}

// Ensure showScreen exists before overriding
if (typeof showScreen === 'function') {
    const originalShowScreen = showScreen;
    showScreen = function(screen) {
        if (screen === 'admin' && !promptAdminPassword()) {
            return;
        }
        originalShowScreen(screen);
    };
} else {
    console.error('showScreen function is not defined');
}