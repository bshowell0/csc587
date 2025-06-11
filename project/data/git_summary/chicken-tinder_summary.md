### Overall Summary

Chicken Tinder is a full-stack web application designed to solve the common social dilemma of a group of friends being unable to decide on a place to eat. It facilitates collaborative decision-making by allowing users to form groups, nominate restaurants, and vote on the options in a quick, "Tinder-style" swiping interface. The project is structured as a monorepo containing three distinct packages: a React frontend, a Node.js/Express backend, and a Cypress testing suite.

The application's theme is humorously centered around chickens. Users are "Hens," groups are "Flocks" or "Coops," members are "Chicks," and restaurant nominations are "Eggs" placed in a "Basket." This playful abstraction makes the user experience more engaging. A key feature is its real-time functionality, powered by WebSockets (Socket.io), which allows all members of a flock to see updates instantly as new members join or restaurants are nominated, without needing to refresh the page.

The backend is built with Node.js and Express, using a MongoDB database (managed via Mongoose) for data persistence. It exposes a RESTful API for managing user authentication (JWT-based), flocks, and the voting process. A unique aspect is the generation of memorable, human-readable codes for flocks (e.g., "zany-waffle") by combining random adjectives and food items from pre-defined lists. The frontend is a single-page application built with React, utilizing React Router for navigation and React Context for state management of authentication and WebSocket communications. Styling is handled with Tailwind CSS. The entire project is configured for continuous integration and deployment, with GitHub Actions workflows to automatically test and deploy the frontend to Netlify and the backend to Azure Web Apps.

### Key Code and Structure Details

**1. Monorepo and Project Structure:**
The project uses `npm` workspaces to manage a monorepo structure located in the `packages/` directory.
-   **`packages/backend`**: Contains the Node.js/Express.js server, Mongoose models, API routes, and WebSocket logic.
-   **`packages/frontend`**: A standard Create React App project containing all UI components, pages, and client-side logic.
-   **`packages/testing`**: Holds the Cypress end-to-end tests.
The root `package.json` defines scripts for running the entire application stack, including a `test` script that uses `start-server-and-test` to spin up the backend and frontend before running Cypress tests.

**2. Backend (Node.js/Express):**

-   **Database Schemas (`packages/backend/flock.js`):** The core data structures are defined here using Mongoose.
    -   `henSchema`: Represents a registered user with `henName`, `email`, and a hashed `password`.
    -   `flockSchema`: The central model for a group session. It includes a unique `coopName`, an `owner` (linking to a `Hen`), an array of `chicks`, a `basket` of `eggs`, and a `step` to track the group's progress through the decision-making flow (1: Lobby, 2: Nominations, 3: Voting, 4: Winner).
    -   `chickSchema`: A subdocument representing a member within a flock, identified by `name`.
    -   `eggSchema`: A subdocument for a nominated restaurant, containing its `title` and running tallies of `yesVotes` and `noVotes`.

-   **Real-time Communication (`packages/backend/index.js`):** The application uses `socket.io` for real-time updates. When a user connects, they join a room specific to their flock's `coopName`.
    ```javascript
    io.on("connection", (socket) => {
        socket.on("join-flock", (code) => {
            socket.join(code);
        });
        // ...
    });
    ```
    When an action modifies the flock's state (e.g., a new user joins or a restaurant is nominated), the server emits a `flock-updated` event to all clients in that room, pushing the new state.
    ```javascript
    // Example from the route that adds a chick
    io.to(req.params.coopName).emit("message", {
        type: "flock-updated",
        newState: chickAndFlock["newFlock"],
    });
    ```

-   **Decision Algorithm (`packages/backend/decision.js`):** The `getWinningRestaurant` function determines the winner. It iterates through the restaurants in the flock's `basket`. The primary criterion is the number of `yesVotes`. In case of a tie, a secondary criterion, the ratio of `yesVotes` to `noVotes`, is used to break the tie. This ensures that a restaurant with, for example, 5 "yes" and 1 "no" vote wins over one with 5 "yes" and 3 "no" votes.

-   **Authentication (`packages/backend/auth.js`):** Authentication is handled via JSON Web Tokens (JWT). The `/auth/login` endpoint validates credentials using `bcrypt.compareSync` and, if successful, signs a JWT containing the user's `henID` and an expiration time. A helper function, `getUserId`, acts as middleware to protect routes by extracting and verifying the token from the `Authorization` header on subsequent requests.

-   **Flock Code Generation (`packages/backend/code-generation/code-generator.js`):** To create user-friendly group codes, this utility reads from `adjectives.txt` and `foods.txt`. It then randomly selects one word from each file to create a unique, hyphenated code (e.g., `happy-taco`). It also handles potential collisions by appending a number if the generated code already exists in the database.

**3. Frontend (React):**

-   **State Management with Context (`packages/frontend/src/context/`):**
    -   **`auth-context.js`**: Manages the user's authentication state. It stores the JWT in a cookie for persistence (`js-cookie`). The `login` function makes an API call and updates the context state, while a `checkToken` function verifies the token's validity with the backend on application load.
    -   **`coop-context.js`**: Manages the `socket.io` client connection. The `connectToFlock` function establishes the connection and joins the appropriate room. It listens for `message` events from the server and updates its state (`lastMessage`, `messages`), making real-time data available to any component that consumes the context.

-   **Application Flow Control (`packages/frontend/src/pages/MainFlockPage.js`):** This component acts as a router for the main user journey within a flock. It fetches the flock's state from the backend and uses the `flock.step` property to conditionally render the appropriate page component: `GroupListPage` (step 1), `NominationPage` (step 2), `VotingPage` (step 3), or `WinnerPage` (step 4). It also subscribes to the `CoopContext` and updates its local `flock` state whenever a `flock-updated` message is received, ensuring the UI reflects the current state in real-time.

-   **Interactive Voting (`packages/frontend/src/pages/VotingPage.js`):** This page implements the core "Tinder-like" voting experience.
    -   It fetches a random, un-voted-on restaurant ("egg") for the current user.
    -   To enhance the experience, it also fetches a relevant GIF from the Tenor API (via a backend endpoint that protects the API key) to display with the restaurant name.
    -   It uses the `react-timer-hook` to implement a 5-second voting timer. If the timer expires, it automatically submits a neutral vote (`postVote(body)` with `egg.vote = 0`). This fast-paced interaction design encourages quick decisions. Once all options are voted on, the user is moved to a loading screen while they await the result.

**4. Testing & CI/CD:**

-   **End-to-End Testing (`packages/testing/cypress/e2e/single-user.cy.js`):** The repository includes a Cypress test that simulates a full "happy path" workflow for a single user acting as the group leader. It covers logging in, creating a flock, getting the unique coop name, nominating several restaurants, and voting on all of them to see the winner page.
-   **CI/CD Workflows (`.github/workflows/`):**
    -   `ci-testing.yml`: Runs on every push and pull request to `main`. It installs dependencies, runs the linter, and executes the Cypress tests (`npm run test`), ensuring code quality and preventing regressions.
    -   `ci-cd_chickentinder-backend.yml`: A deploy workflow for the backend. On a push to `main`, it builds the Node.js app, zips it as an artifact, and deploys it to a pre-configured Azure Web App.
    -   `frontend-deploy.yml`: A similar workflow for the frontend, which builds the React app and deploys the production files to Netlify.
