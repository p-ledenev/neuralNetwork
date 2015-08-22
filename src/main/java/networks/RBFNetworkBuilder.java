package networks;

import lombok.*;
import org.encog.mathutil.rbf.*;
import org.encog.ml.*;
import org.encog.ml.data.*;
import org.encog.neural.rbf.*;

/**
 * Created by ledenev.p on 17.08.2015.
 */

@NoArgsConstructor
@Data
public class RBFNetworkBuilder extends NetworkBuilder {

    private int hiddenLayerSize = 100;
    private RBFEnum networkType = RBFEnum.Gaussian;

    @Override
    protected BasicML createNetwork(MLDataSet trainingSet) {
        return new RBFNetwork(trainingSet.getInputSize(), hiddenLayerSize, trainingSet.getIdealSize(), networkType);
    }

    protected String getName() {
        return "RBFNetwork";
    }
}
