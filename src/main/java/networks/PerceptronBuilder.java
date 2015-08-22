package networks;

import org.encog.engine.network.activation.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.neural.networks.*;
import org.encog.neural.networks.layers.*;

/**
 * Created by ledenev.p on 18.08.2015.
 */

public class PerceptronBuilder extends NetworkBuilder {

    @Override
    protected String getName() {
        return "Perceptron";
    }

    @Override
    protected BasicML createNetwork(MLDataSet trainingSet) {

        BasicNetwork network;
        //network = treeLayersNetwork(trainingSet);
        //network = twoLayersNetwork(trainingSet);
        network = fourLayersNetwork(trainingSet);

        network.getStructure().finalizeStructure();
        network.reset();

        return network;
    }

    private BasicNetwork treeLayersNetwork(MLDataSet trainingSet) {

        BasicNetwork network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationTANH(), true, trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 75));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 25));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, trainingSet.getIdealSize()));

        return network;
    }

    private BasicNetwork twoLayersNetwork(MLDataSet trainingSet) {

        BasicNetwork network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationTANH(), true, trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 75));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, trainingSet.getIdealSize()));

        return network;
    }

    private BasicNetwork fourLayersNetwork(MLDataSet trainingSet) {

        BasicNetwork network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationTANH(), true, trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, 50));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, trainingSet.getIdealSize()));

        return network;
    }
}
