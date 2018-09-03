from unittest import TestCase

import torch

from models.caffenet import caffenet, AlexNetCaffe


class TestCaffenet(TestCase):
    def test_caffenet(self):
        self.assertEqual(type(caffenet(100)), type(AlexNetCaffe(100)))

    def test_get_params(self):
        self.assertIsNotNone(caffenet(100).get_params())

    def test_forward(self):
        n_classes = 100
        batch_size = 4
        net = caffenet(n_classes)
        X = torch.zeros(batch_size, 3, 227, 227)
        y = net(X)
        self.assertEquals(list(y.shape), [batch_size, n_classes])

