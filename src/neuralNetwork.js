import {
  Array1D,
  InCPUMemoryShuffledInputProviderBuilder,
  Graph,
  Session,
  SGDOptimizer,
  ENV,
  NDArrayMath,
  CostReduction,
} from 'deeplearn';

class MnistModel {
  math = ENV.math;

  session;

  initialLearningRate = 0.06;
  optimizer;

  batchSize = 300;

  inputTensor;
  targetTensor;
  costTensor;
  predictionTensor;

  feedEntries;

  constructor() {
    this.optimizer = new SGDOptimizer(this.initialLearningRate);
  }

  setupSession(trainingSet) {
    const graph = new Graph();

    this.inputTensor = graph.placeholder('input unrolled pixels', [784]);
    this.targetTensor = graph.placeholder('output digit classifier', [10]);

    let fullyConnectedLayer = this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);
    fullyConnectedLayer = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);

    this.predictionTensor = this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 10);
    this.costTensor = graph.meanSquaredCost(this.targetTensor, this.predictionTensor);

    this.session = new Session(graph, this.math);

    this.prepareTrainingSet(trainingSet);
  }

  prepareTrainingSet(trainingSet) {
    const oldMath = ENV.math;
    const safeMode = false;
    const math = new NDArrayMath('cpu', safeMode);
    ENV.setMath(math);

    const inputArray = trainingSet.map(v => Array1D.new(v.input));
    const targetArray = trainingSet.map(v => Array1D.new(v.output));

    const shuffledInputProviderBuilder = new InCPUMemoryShuffledInputProviderBuilder([ inputArray, targetArray ]);
    const [ inputProvider, targetProvider ] = shuffledInputProviderBuilder.getInputProviders();

    this.feedEntries = [
      { tensor: this.inputTensor, data: inputProvider },
      { tensor: this.targetTensor, data: targetProvider },
    ];

    ENV.setMath(oldMath);
  }

  train(step, computeCost) {
    let learningRate = this.initialLearningRate * Math.pow(0.90, Math.floor(step / 50));
    this.optimizer.setLearningRate(learningRate);

    let costValue;
    this.math.scope(() => {
      const cost = this.session.train(
        this.costTensor,
        this.feedEntries,
        this.batchSize,
        this.optimizer,
        computeCost ? CostReduction.MEAN : CostReduction.NONE,
      );

      if (computeCost) {
        costValue = cost.get();
      }
    });

    return costValue;
  }

  predict(pixels) {
    let classifier = [];

    this.math.scope(() => {
      const mapping = [{
        tensor: this.inputTensor,
        data: Array1D.new(pixels),
      }];

      classifier = this.session.eval(this.predictionTensor, mapping).getValues();
    });

    return [ ...classifier ];
  }

  createFullyConnectedLayer(
    graph,
    inputLayer,
    layerIndex,
    units,
    activationFunction
  ) {
    return graph.layers.dense(
      `fully_connected_${layerIndex}`,
      inputLayer,
      units,
      activationFunction
        ? activationFunction
        : (x) => graph.relu(x)
    );
  }
}

export default MnistModel;