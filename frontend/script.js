document.addEventListener('DOMContentLoaded', () => {
    // 1. Pit Entry Countdown
    const countdownDisplay = document.getElementById('pitCountdown');
    let timeLeft = 12; // Starting at 12 seconds as per design

    function updateCountdown() {
        const minutes = String(Math.floor(timeLeft / 60)).padStart(2, '0');
        const seconds = String(timeLeft % 60).padStart(2, '0');
        countdownDisplay.textContent = `${minutes}:${seconds}`;

        if (timeLeft > 0) {
            timeLeft--;
        } else {
            // Optional: Actions when countdown finishes
            // clearInterval(countdownInterval);
            // countdownDisplay.textContent = "00:00";
            // console.log("Pit entry countdown finished!");
        }
    }

    // Update countdown every second
    const countdownInterval = setInterval(updateCountdown, 1000);
    // Call immediately to show initial value
    updateCountdown();

    // 2. Rival Boxing Likelihood - Progress Bars and Colors
    const progressBars = document.querySelectorAll('.progress-bar');

    progressBars.forEach(bar => {
        const likelihood = parseInt(bar.dataset.likelihood);
        bar.style.width = `${likelihood}%`; // Set width dynamically

        // Apply color based on likelihood
        if (likelihood >= 80) {
            bar.classList.add('green');
        } else if (likelihood >= 50) {
            bar.classList.add('yellow');
        } else {
            bar.classList.add('red');
        }
    });

    // 3. Safe Pit Button (Example interaction)
    const safePitButton = document.querySelector('.safe-pit-button');
    safePitButton.addEventListener('click', () => {
        alert('Safe Pit strategy confirmed!'); // Replace with actual functionality
        // You might want to:
        // - Send data to a backend
        // - Update other parts of the dashboard
        // - Trigger an animation
    });

    // Optional: Example of updating data dynamically (e.g., from an API)
    // You would typically fetch real data here and update elements.
    function updateDashboardData() {
        // Example: Update Red Bull's likelihood
        // const redBullBar = document.querySelector('.team-item:first-child .progress-bar');
        // const newLikelihood = Math.floor(Math.random() * 100);
        // redBullBar.dataset.likelihood = newLikelihood;
        // redBullBar.style.width = `${newLikelihood}%`;
        //
        // // Reapply colors
        // redBullBar.classList.remove('red', 'yellow', 'green');
        // if (newLikelihood >= 80) {
        //     redBullBar.classList.add('green');
        // } else if (newLikelihood >= 50) {
        //     redBullBar.classList.add('yellow');
        // } else {
        //     redBullBar.classList.add('red');
        // }
        // document.querySelector('.team-item:first-child .likelihood-value').textContent = `${newLikelihood}%`;

        // Example: Update lap info
        // document.querySelector('.current-info-panel .info-item:first-child h3').textContent = `CURRENT LAP: ${Math.floor(Math.random() * 70)}/70`;
    }

    // You could call updateDashboardData periodically with setInterval
    // setInterval(updateDashboardData, 10000); // Every 10 seconds for example
});