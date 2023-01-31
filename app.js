const express = require('express');
const bodyParser = require('body-parser');
const TensorFlow = require('@tensorflow/tfjs-node');

const app = express();
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Serve the HTML file
app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

// Endpoint for receiving the user question
app.post('/recommend', (req, res) => {
  const question = req.body.question;

  // Call the function to get the recommended questions
  const recommendedQuestions = recommendQuestions(question);

  // Send the recommended questions back to the frontend
  res.send(recommendedQuestions);
});

// Function to get the recommended questions
async function recommendQuestions(question) {
  // Your code for finding similar questions here
  // ...

  return recommendedQuestions;
}

// Start the server
app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
