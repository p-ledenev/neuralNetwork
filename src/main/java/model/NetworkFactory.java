package model;

import org.encog.engine.network.activation.ActivationTANH;
import org.encog.mathutil.rbf.RBFEnum;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.rbf.RBFNetwork;

/**
 * Created by DiKey on 17.08.2015.
 */
public class NetworkFactory {

    public static RBFNetwork rbfNetwork(MLDataSet trainingSet, int hiddenLayerSize, RBFEnum type) {
        return new RBFNetwork(trainingSet.getInputSize(), hiddenLayerSize, trainingSet.getIdealSize(), type);
    }

    public static BasicNetwork basicNetwork(MLDataSet trainingSet, int hiddenLayerSize) {

        BasicNetwork network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationTANH(), true, trainingSet.getInputSize()));
        network.addLayer(new BasicLayer(new ActivationTANH(), true, hiddenLayerSize));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, trainingSet.getIdealSize()));

        network.getStructure().finalizeStructure();
        network.reset();

        return network;
    }
}
