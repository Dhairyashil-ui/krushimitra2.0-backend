const request = require('supertest');
let app;

beforeAll(() => {
  // Lazy require to ensure environment variables loaded
  app = require('../server').app || require('../server');
});

describe('GET /training/metrics', () => {
  it('returns metrics object', async () => {
    const res = await request(app).get('/training/metrics');
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('data');
    expect(res.body.data).toHaveProperty('model');
  });
});
