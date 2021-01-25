import unittest

import tensorflow as tf

import efficientdet.data.voc as voc
import efficientdet.models as models


class EfficientDetTest(unittest.TestCase):

    def _compare_shapes(self, shape1, shape2):
        self.assertTrue(len(shape1) == len(shape2))

        for s1, s2 in zip(shape1, shape2):
            if s1 is not None and s2 is not None:
                self.assertEqual(s1, s2)

    # def test_forward(self):
    #     batch_size = 2
    #     num_classes = len(voc.IDX_2_LABEL)
    #     model = models.EfficientDet(num_classes=num_classes,
    #                                 D=0,
    #                                 weights=None)

    #     input_size = model.config.input_size

    #     ds = voc.build_dataset('test/data/VOC2007',
    #                            batch_size=batch_size,
    #                            im_input_size=(input_size, input_size))

    #     for images, annotations in ds.take(1):
    #         bb, clf = model([images, annotations], training=True)

    #     self._compare_shapes(bb.shape, [batch_size, None, 9, 4])
    #     self._compare_shapes(clf.shape, [batch_size, None, 9, num_classes])
        
    def test_forward_inference(self):
        batch_size = 2
        num_classes = len(voc.IDX_2_LABEL)
        model = models.EfficientDet(num_classes=num_classes,
                                    D=0,
                                    weights=None)

        input_size = model.config.input_size

        ds = voc.build_dataset('test/data/VOC2007', im_input_size=input_size)
        images, annotations = next(iter(ds.take(1)))
        boxes, labels, scores = model(tf.expand_dims(images, 0), training=False)
        
        # TODO: migrate to tensors when output is padded
        for i in range(len(boxes)):
            self._compare_shapes(boxes[i].shape, [None, 4])
            self._compare_shapes(labels[i].shape, [None])

        
if __name__ == "__main__":
    unittest.main()
