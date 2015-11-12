import model.*;
import org.encog.*;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.*;
import org.encog.ml.data.basic.*;
import org.encog.ml.data.specific.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.*;
import org.encog.neural.networks.training.propagation.resilient.*;
import org.encog.util.csv.*;
import org.junit.*;
import org.junit.Test;

import java.io.*;
import java.util.*;

/**
 * Created by ledenev.p on 10.11.2015.
 */
public class NetworkTests {

    @Test
    public void simpleClassification() throws Throwable {

        // if key > 0.5 than 1 else 0
        MLDataSet trainingSet = new CSVNeuralDataSet(Runner.dataPath + "/testNetworkTrainingData.csv", 1, 1, false, new CSVFormat('.', ';'), false);
        MLDataSet testSet = new CSVNeuralDataSet(Runner.dataPath + "/testNetworkTestData.csv", 1, 1, false, new CSVFormat('.', ';'), false);

        BasicNetwork network = createNetwork(trainingSet);
        ResilientPropagation training = new ResilientPropagation(network, trainingSet);
        training.setError(1.);

        Log.info("Starting training");
        for (int i = 1; training.getError() > 0.001; i++) {
            training.iteration();

            if (i % 10 == 0)
                printIteration(i, training, testSet);
        }

        training.finishTraining();
        Encog.getInstance().shutdown();

        printIteration(-1, training, testSet);
        //printTestResults(network, testSet);
        //printNetwork(network);
    }

    private void printIteration(int i, ResilientPropagation training, MLDataSet testSet) {
        Log.info("Iteration: " + i + "; training error: " + Round.toAmount(training.getError(), 6) +
                "; test error: " + Round.toAmount(training.getCurrentFlatNetwork().calculateError(testSet), 6));
    }

    private BasicNetwork createNetwork(MLDataSet trainingSet) {

        BasicNetwork network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationTANH(), true, trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 10));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, trainingSet.getIdealSize()));

        network.getStructure().finalizeStructure();
        network.reset();

        return network;
    }

    private void printTestResults(BasicNetwork network, MLDataSet testSet) throws Throwable {

        int i = 0;
        for (MLDataPair pair : testSet) {
            MLDataSet set = new BasicMLDataSet();
            set.add(pair);

            MLData output = network.compute(pair.getInput());

            Log.info("input: " + pair.getInput().getData(0) + "; tutor value: " + pair.getIdeal().getData(0) + "; Network value: " +
                    Round.toAmount(output.getData(0), 6) +
                    "; totalNetworkError: " + Round.toAmount(network.calculateError(set), 6));
        }
    }

    private void printNetwork(BasicNetwork network) {

        final StringBuilder result = new StringBuilder();
        result.append("\n");

        for (int layer = 0; layer < network.getLayerCount() - 1; layer++) {

            boolean hasBias = network.isLayerBiased(layer);

            for (int fromNeuronIndex = 0; fromNeuronIndex < network.getLayerNeuronCount(layer) + (hasBias ? 1 : 0); fromNeuronIndex++) {
                for (int toNeuronIndex = 0; toNeuronIndex < network.getLayerNeuronCount(layer + 1); toNeuronIndex++) {
                    String fromNeuron = "hide [" + (layer) + ",";
                    String toNeuron = "hide [" + (layer + 1) + ",";

                    if (layer == 0)
                        fromNeuron = "inpt [0,";

                    if (hasBias && (fromNeuronIndex == network.getLayerNeuronCount(layer)))
                        fromNeuron = "bias [" + (layer) + ",";

                    if (layer == (network.getLayerCount() - 2))
                        toNeuron = "outp [" + (network.getLayerCount() - 1) + ",";

                    fromNeuron = fromNeuron + fromNeuronIndex;

                    result.append(fromNeuron + "] --> " + toNeuron + toNeuronIndex
                            + "] : " + Round.toAmount(network.getWeight(layer, fromNeuronIndex, toNeuronIndex), 6)
                            + "\n");
                }
            }
        }

        Log.info(result.toString());
    }

    @Ignore
    @Test
    public void generateData() throws Throwable {

        PrintWriter writer = new PrintWriter(Runner.dataPath + "/testNetworkTestData.csv", "UTF-8");

        Random random = new Random();
        for (int i = 0; i < 1000; i++) {
            Double input = random.nextInt(10000) / 10000.;
            writer.println(input + ";" + (input > 0.5 ? "1" : "0"));
        }

        writer.close();
    }
}
