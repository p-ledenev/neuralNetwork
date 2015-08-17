package model;

import lombok.*;
import org.encog.mathutil.rbf.*;
import org.encog.ml.data.*;
import org.encog.neural.rbf.*;
import org.encog.neural.rbf.training.*;

/**
 * Created by ledenev.p on 17.08.2015.
 */

@NoArgsConstructor
public class RBFNetworkBuilder {

    @Getter
    private RBFNetwork network;

    public void build((MLDataSet trainingSet, int hiddenLayerSize, RBFEnum type) {
        network = new RBFNetwork(trainingSet.getInputSize(), hiddenLayerSize, trainingSet.getIdealSize(), type);
    }
}
