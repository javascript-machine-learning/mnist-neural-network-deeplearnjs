import React, { Component } from 'react';
import mnist from './mnist';

import './App.css';

import MnistModel from './neuralNetwork';

const ITERATIONS = 750;
const TRAINING_SET_SIZE = 3000;
const TEST_SET_SIZE = 50;

class App extends Component {

  testSet;
  trainingSet;
  mnistModel;

  constructor() {
    super();
    const { training, test } = mnist.set(TRAINING_SET_SIZE, TEST_SET_SIZE);

    this.testSet = test;
    this.trainingSet = training;

    this.mnistModel = new MnistModel();
    this.mnistModel.setupSession(this.trainingSet);

    this.state = {
      currentIteration: 0,
      cost: -1,
    };
  }

  componentDidMount () {
    requestAnimationFrame(this.tick);
  };

  tick = () => {
    this.setState((state) => ({ currentIteration: state.currentIteration + 1 }));

    if (this.state.currentIteration < ITERATIONS) {
      requestAnimationFrame(this.tick);

      let computeCost = !(this.state.currentIteration % 5);
      let cost = this.mnistModel.train(this.state.currentIteration, computeCost);

      if (cost > 0) {
        this.setState(() => ({ cost }));
      }
    }
  };

  render() {
    const { currentIteration, cost } = this.state;
    return (
      <div className="app">
        <div>
          <h1>Neural Network for MNIST Digit Recognition in JavaScript</h1>
          <p><strong>Iterations:</strong> {currentIteration}</p>
          <p><strong>Cost:</strong> {cost.toFixed(3)}</p>
        </div>

        <TestExamples
          model={this.mnistModel}
          testSet={this.testSet}
        />
      </div>
    );
  }
}

const TestExamples = ({ model, testSet }) =>
  <div className="test-example-list">
    {Array(TEST_SET_SIZE).fill(0).map((v, i) =>
      <TestExampleItem
        key={i}
        model={model}
        input={testSet[i].input}
        output={testSet[i].output}
      />
    )}
  </div>

const TestExampleItem = ({ model, input, output }) =>
  <div className="test-example-item">
    <MnistDigit
      digitInput={input}
    />

    <PredictedMnistDigit
      digitInput={model.predict(input)}
      digitOutput={output}
    />
  </div>

class MnistDigit extends Component {
  shouldComponentUpdate() {
    return false;
  }

  render() {
    const { digitInput } = this.props;
    return (
      <div>
        {fromUnrolledToPartition(digitInput, 28).map((row, i) =>
          <div key={i} className="pixel-row">
            {row.map((p, j) =>
              <div
                key={j}
                className="pixel"
                style={{ backgroundColor: denormalizeAndColorize(p) }}
              />
            )}
          </div>
        )}
      </div>
    );
  }
}

const PredictedMnistDigit = ({ digitInput, digitOutput }) => {
  const digit = fromClassifierToDigit(digitInput);

  return (
    <div
      className="test-example-item-prediction"
      style={getColor(digitOutput, digit)}
    >
      <div className="prediction-digit">
        {digit.number}
      </div>
      <div className="prediction-probability">
        p(x)={digit.probability.toFixed(2)}
      </div>
    </div>
  );
}

const getColor = (output, digit) =>
  fromClassifierToDigit(output).number === digit.number
    ? { backgroundColor: '#55AA55' }
    : { backgroundColor: '#D46A6A' }

const fromClassifierToDigit = (classifier) =>
  classifier.reduce(toNumber, { number: -1, probability: -1 });

const toNumber = (result, value, key) => {
  if (value > result.probability) {
    result = { number: key, probability: value };
  }
  return result;
};

const fromUnrolledToPartition = (digit, size) =>
  digit.reduce(toPartition(size), []);

const toPartition = (size) => (result, value, key) => {
  if (key % size === 0) {
    result.push([]);
  }

  result[result.length - 1].push(value);

  return result;
};

const denormalizeAndColorize = (p) =>
  compose(
    toColor,
    denormalize
  )(p);

const denormalize = (p) =>
  (p * 255).toFixed(0);

const toColor = (colorChannel) =>
  `rgb(${colorChannel}, ${colorChannel}, ${colorChannel})`;

const compose = (...fns) =>
  fns.reduce((f, g) => (...args) => f(g(...args)));

export default App;
